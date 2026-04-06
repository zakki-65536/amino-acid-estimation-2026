# compare_catboost_lightgbm_by_id_random_subsets.py
# ------------------------------------------------------------
# Excelをpandasで読み込み：
# - 1行目: 項目名（先頭 c=カテゴリ, n=数値）
# - 2行目: 項目ID（学習時の列名として使用）
# - 3行目以降: データ
# - 最終列が目的変数（回帰）
#
# 動作：
# - 説明変数(全feature)から「毎回ランダムに」列数も列の選択も変えてサブセットを作り
# - CatBoost / LightGBM を同一KFoldで評価
# - 無限に繰り返す（Ctrl+Cで停止）
#
# 各実行ごとにコンソールへ1行出力：
#   timestamp(mm/dd hh:mm:ss) CB_RMSE_mean CB_RMSE_std CB_MAE_mean CB_MAE_std LG_RMSE_mean LG_RMSE_std LG_MAE_mean LG_MAE_std
#
# さらに、全フォールド平均の特徴量重要度（寄与率%）を
#   result_catboost.xlsx / result_lightgbm.xlsx
# に「新しいシート名(mmddhhmmss)」として追記保存する。
# そのときのシート名時刻はコンソールのtimestampと一致する。
#
# 必要:
#   pip install pandas openpyxl scikit-learn catboost lightgbm
# ------------------------------------------------------------

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

from catboost import CatBoostRegressor
import lightgbm as lgb


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def make_unique_names(names: list[str]) -> list[str]:
    """重複列名があると学習で事故るので、重複時は _2, _3... を付与してユニーク化"""
    seen = {}
    out = []
    for n in names:
        n = str(n)
        if n not in seen:
            seen[n] = 1
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
    return out


def to_contrib_df(features: list[str], importances: np.ndarray) -> pd.DataFrame:
    """重要度から寄与率(%)を作ってDataFrame化（合計が0のときは0%）"""
    imp = np.asarray(importances, dtype=float)
    total = float(np.sum(imp))
    if total == 0.0 or np.isnan(total):
        contrib = np.zeros_like(imp)
    else:
        contrib = imp / total * 100.0

    df = pd.DataFrame({"Feature": features, "Importance": imp, "Contribution(%)": contrib})
    df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
    return df


def append_df_to_excel(filepath: str, sheet_name: str, df: pd.DataFrame) -> None:
    """Excelに新規シートとして追記保存（無ければ新規作成）"""
    p = Path(filepath)
    if p.exists():
        with pd.ExcelWriter(filepath, engine="openpyxl", mode="a", if_sheet_exists="new") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True, help="Excelファイルパス (.xlsx)")
    parser.add_argument("--sheet", default=0, help="シート名またはインデックス（既定: 0）")
    parser.add_argument("--n_splits", type=int, default=5, help="CV分割数（既定: 5）")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード（既定: 42）")
    parser.add_argument("--min_features", type=int, default=5, help="抽出する特徴量の最小数（既定: 5）")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ------------------------
    # 1) Excelをそのまま読み込み（ヘッダ無しで読み込む）
    # ------------------------
    raw = pd.read_excel(args.excel, sheet_name=args.sheet, header=None, engine="openpyxl")

    if raw.shape[0] < 3:
        raise ValueError("行数が足りません（最低でも1行目=項目名, 2行目=項目ID, 3行目以降=データが必要）")
    if raw.shape[1] < 2:
        raise ValueError("列数が足りません（説明変数＋目的変数が必要）")

    # 1行目（index=0）: 項目名  / 2行目（index=1）: 項目ID
    item_names = raw.iloc[0, :].astype(str).tolist()
    item_ids = raw.iloc[1, :].tolist()

    # 列名（学習用）は項目IDを使用（数字でもOKだが、扱いを安定させるため文字列化）
    col_ids = [str(x) for x in item_ids]
    col_ids = make_unique_names(col_ids)  # 念のため重複回避

    # 3行目以降がデータ
    df = raw.iloc[2:, :].copy()
    df.columns = col_ids
    df.reset_index(drop=True, inplace=True)

    # 最終列が目的変数
    target_col = df.columns[-1]
    all_feature_cols = list(df.columns[:-1])

    # ------------------------
    # 2) c/n 判定は「項目名（1行目）」で行い、列名は「項目ID」で扱う
    # ------------------------
    name_by_id = dict(zip(col_ids, item_names))
    all_cat_cols = [cid for cid in all_feature_cols if name_by_id.get(cid, "").strip().lower().startswith("c")]
    all_num_cols = [cid for cid in all_feature_cols if name_by_id.get(cid, "").strip().lower().startswith("n")]

    other_cols = [cid for cid in all_feature_cols if cid not in all_cat_cols and cid not in all_num_cols]
    if other_cols:
        # これは「毎回」出ると鬱陶しいので一度だけ出す
        print(
            f"[WARN] 項目名が c/n で始まらない列が {len(other_cols)} 個あります。"
            f"数値扱いに寄せます（ID）: {other_cols[:10]}"
        )
        all_num_cols += other_cols

    # ------------------------
    # 3) 目的変数を数値化、説明変数を準備
    # ------------------------
    y = pd.to_numeric(df[target_col], errors="coerce")

    # 欠損のある目的変数行は落とす（学習不能なので）
    valid_idx = y.notna()
    if valid_idx.sum() != len(y):
        print(f"[WARN] 目的変数がNaNの行を除外します: {len(y) - valid_idx.sum()} 行")
        df = df.loc[valid_idx].reset_index(drop=True)
        y = y.loc[valid_idx].reset_index(drop=True)

    X_all = df[all_feature_cols].copy()

    # 数値列を数値化（変換不可はNaN）
    for c in all_num_cols:
        X_all[c] = pd.to_numeric(X_all[c], errors="coerce")

    # ------------------------
    # 4) 無限ループでランダムサブセット評価
    # ------------------------
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    n_total_features = len(all_feature_cols)
    min_k = max(1, min(args.min_features, n_total_features))

    while True:
        # タイムスタンプ（コンソールとシート名で同一時刻を使う）
        now = datetime.now()
        ts_console = now.strftime("%m/%d %H:%M:%S")
        ts_sheet = now.strftime("%m%d%H%M%S")

        # ---- ランダムに特徴量サブセットを作る（列数も列の選択も毎回ランダム） ----
        k = int(rng.integers(low=min_k, high=n_total_features + 1))
        chosen = rng.choice(all_feature_cols, size=k, replace=False).tolist()

        cat_cols = [c for c in chosen if c in all_cat_cols]
        num_cols = [c for c in chosen if c in all_num_cols]  # 使わないが念のため保持

        # サブセットのXを作る
        X_sub = X_all[chosen].copy()

        # CatBoost用：カテゴリは文字列
        X_cb = X_sub.copy()
        for c in cat_cols:
            X_cb[c] = X_cb[c].where(~X_cb[c].isna(), "missing").astype(str)

        # LightGBM用：カテゴリはcategory dtype
        X_lgb = X_sub.copy()
        for c in cat_cols:
            X_lgb[c] = X_lgb[c].where(~X_lgb[c].isna(), "missing").astype(str).astype("category")

        cat_feature_indices = [X_cb.columns.get_loc(c) for c in cat_cols]

        cb_rmses, cb_maes = [], []
        lgb_rmses, lgb_maes = [], []

        cb_importances = []
        lgb_importances = []

        # ---- CV ----
        for tr_idx, va_idx in kf.split(X_cb):
            X_tr_cb, X_va_cb = X_cb.iloc[tr_idx], X_cb.iloc[va_idx]
            X_tr_lgb, X_va_lgb = X_lgb.iloc[tr_idx], X_lgb.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            # CatBoost
            cb = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=10000,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=10,
                random_seed=args.seed,
                allow_writing_files=False,
                verbose=False,
            )
            cb.fit(
                X_tr_cb, y_tr,
                cat_features=cat_feature_indices,
                eval_set=(X_va_cb, y_va),
                early_stopping_rounds=200,
                use_best_model=True,
            )
            pred_cb = cb.predict(X_va_cb)
            cb_rmses.append(rmse(y_va, pred_cb))
            cb_maes.append(mae(y_va, pred_cb))
            cb_importances.append(cb.get_feature_importance())

            # LightGBM（Info非表示）
            lgbm = lgb.LGBMRegressor(
                n_estimators=10000,
                learning_rate=0.03,
                num_leaves=64,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=args.seed,
                n_jobs=-1,
                verbosity=-1,
            )
            lgbm.fit(
                X_tr_lgb, y_tr,
                eval_set=[(X_va_lgb, y_va)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
                categorical_feature=cat_cols,
            )
            pred_lgb = lgbm.predict(X_va_lgb, num_iteration=lgbm.best_iteration_)
            lgb_rmses.append(rmse(y_va, pred_lgb))
            lgb_maes.append(mae(y_va, pred_lgb))
            lgb_importances.append(lgbm.booster_.feature_importance(importance_type="gain"))

        # ---- 集計 ----
        cb_rmse_mean, cb_rmse_std = float(np.mean(cb_rmses)), float(np.std(cb_rmses))
        cb_mae_mean, cb_mae_std = float(np.mean(cb_maes)), float(np.std(cb_maes))
        lg_rmse_mean, lg_rmse_std = float(np.mean(lgb_rmses)), float(np.std(lgb_rmses))
        lg_mae_mean, lg_mae_std = float(np.mean(lgb_maes)), float(np.std(lgb_maes))

        # ---- 指定フォーマットで1行出力（スペース区切り）----
        print(
            f"{ts_console} "
            f"{cb_rmse_mean:.6f} {cb_rmse_std:.6f} {cb_mae_mean:.6f} {cb_mae_std:.6f} "
            f"{lg_rmse_mean:.6f} {lg_rmse_std:.6f} {lg_mae_mean:.6f} {lg_mae_std:.6f}"
        )

        # ---- 全フォールド平均の寄与率(%)を作ってExcelへ追記 ----
        mean_cb_imp = np.mean(np.asarray(cb_importances), axis=0)
        mean_lgb_imp = np.mean(np.asarray(lgb_importances), axis=0)

        cb_imp_df = to_contrib_df(list(X_cb.columns), mean_cb_imp)
        lgb_imp_df = to_contrib_df(list(X_lgb.columns), mean_lgb_imp)

        # どの特徴量サブセットで出た結果か追跡したい場合の補助列（邪魔なら消してOK）
        cb_imp_df.insert(0, "Timestamp", ts_console)
        cb_imp_df.insert(1, "NumFeatures", k)
        lgb_imp_df.insert(0, "Timestamp", ts_console)
        lgb_imp_df.insert(1, "NumFeatures", k)

        append_df_to_excel("result_catboost_rr.xlsx", ts_sheet, cb_imp_df)
        append_df_to_excel("result_lightgbm_rr.xlsx", ts_sheet, lgb_imp_df)


if __name__ == "__main__":
    main()