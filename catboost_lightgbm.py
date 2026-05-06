# pip install pandas openpyxl scikit-learn catboost lightgbm
# python catboost_lightgbm.py --data data_204項目_male_空腹時.xlsx --result male_空腹時

# ------------------------------------------------------------
# Excelをpandasで読み込み：
# - 1行目: 項目名（先頭 c=カテゴリ, n=数値）
# - 2行目: 項目ID（学習時の列名として使用）
# - 3行目以降: データ
# - 最終列が目的変数（回帰）
#
# CatBoost と LightGBM を同一のKFoldで比較（RMSE/MAE）
# + 全フォールド平均の特徴量重要度（寄与率%）を出力（CSV保存も）
#
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

from catboost import CatBoostRegressor
import lightgbm as lgb

from datetime import datetime
from pathlib import Path


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def tolerance_accuracy(y_true, y_pred, tolerance_ratio: float = 0.05) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    abs_error = np.abs(y_pred - y_true)
    tolerance = np.abs(y_true) * tolerance_ratio
    correct = abs_error <= tolerance

    return float(np.mean(correct))

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


def main(run_seed: int | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Excelファイルパス (.xlsx)")
    parser.add_argument("--result", required=True, help="結果を保存するExcelファイル名")
    parser.add_argument("--sheet", default=0, help="シート名またはインデックス（既定: 0）")
    parser.add_argument("--n_splits", type=int, default=5, help="CV分割数（既定: 5）")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード（既定: 42）")
    parser.add_argument("--top_k", type=int, default=30, help="重要度上位の表示件数（既定: 30）")
    args = parser.parse_args()

    if run_seed is None:
        run_seed = args.seed

    # runごとに乱数を変える
    rng = np.random.default_rng(run_seed)

    # ------------------------
    # 1) Excelをそのまま読み込み（ヘッダ無しで読み込む）
    # ------------------------
    raw = pd.read_excel(args.data, sheet_name=args.sheet, header=None, engine="openpyxl")

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
    feature_cols = list(df.columns[:-1])

    # ------------------------
    # 2) c/n 判定は「項目名（1行目）」で行い、列名は「項目ID」で扱う
    # ------------------------
    name_by_id = dict(zip(col_ids, item_names))

    cat_cols = [cid for cid in feature_cols if name_by_id.get(cid, "").strip().lower().startswith("c")]
    num_cols = [cid for cid in feature_cols if name_by_id.get(cid, "").strip().lower().startswith("n")]

    other_cols = [cid for cid in feature_cols if cid not in cat_cols and cid not in num_cols]
    if other_cols:
        print(
            f"[WARN] 項目名が c/n で始まらない列が {len(other_cols)} 個あります。"
            f"数値扱いに寄せます（ID）: {other_cols[:10]}"
        )
        num_cols += other_cols

    # ------------------------
    # 3) 目的変数を数値化
    # ------------------------
    y = pd.to_numeric(df[target_col], errors="coerce")

    # 説明変数
    X = df[feature_cols].copy()

    # 数値列を数値化（変換不可はNaN）
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # CatBoost用：カテゴリ列は「必ず文字列」に（実数が混ざってもOKにするため）
    X_cb = X.copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].where(~X_cb[c].isna(), "missing")
        X_cb[c] = X_cb[c].astype(str)

    # LightGBM用：カテゴリ列は category dtype
    X_lgb = X.copy()
    for c in cat_cols:
        X_lgb[c] = X_lgb[c].where(~X_lgb[c].isna(), "missing").astype(str).astype("category")

    # CatBoostのcat_featuresは「列インデックス」
    cat_feature_indices = [X_cb.columns.get_loc(c) for c in cat_cols]

    '''
    print("---- Data Summary ----")
    print(f"Rows (data): {len(df)}")
    print(f"Features: {len(feature_cols)}")
    print(f"  Numeric: {len(num_cols)}")
    print(f"  Categorical: {len(cat_cols)}")
    print(f"Target (last col ID): {target_col}")
    print("----------------------")
    '''

    # 欠損のある目的変数行は落とす（学習不能なので）
    valid_idx = y.notna()
    if valid_idx.sum() != len(y):
        print(f"[WARN] 目的変数がNaNの行を除外します: {len(y) - valid_idx.sum()} 行")
        y = y[valid_idx].reset_index(drop=True)
        X_cb = X_cb.loc[valid_idx].reset_index(drop=True)
        X_lgb = X_lgb.loc[valid_idx].reset_index(drop=True)

    # ------------------------
    # 4) CVで比較（同一分割）
    # ------------------------
    
    # kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    
    # KFoldもrunごとに変える（ここが重要）
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=run_seed)

    cb_rmses, cb_maes = [], []
    lgb_rmses, lgb_maes = [], []

    # ★追加：全フォールド重要度を保存
    cb_importances = []
    lgb_importances = []

    # 正解率計算用に全foldの予測と真値を保持
    cb_all_true_list, cb_all_pred_list = [], []
    lgb_all_true_list, lgb_all_pred_list = [], []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb), start=1):
        X_tr_cb, X_va_cb = X_cb.iloc[tr_idx], X_cb.iloc[va_idx]
        X_tr_lgb, X_va_lgb = X_lgb.iloc[tr_idx], X_lgb.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # ---- CatBoost ----
        cb = CatBoostRegressor(
            loss_function="RMSE",  # 学習時に最小化する目的関数
            eval_metric="RMSE",  # 検証データで見る指標 early stoppingの基準
            iterations=10000,  # 最大ツリー数(ブースティング回数)
            learning_rate=0.03,  # 1本の木がどれだけ強く補正するか 0.03はやや学習が遅め
            depth=6,  # 1本の木の深さの最大 深いほど過学習しやすい
            l2_leaf_reg=10,  # L2正則化（過学習抑制）
            random_seed=run_seed,
            allow_writing_files=False,
            verbose=False,
        )
        cb.fit(
            X_tr_cb, y_tr,
            cat_features=cat_feature_indices,
            eval_set=(X_va_cb, y_va),
            early_stopping_rounds=200,  # 200回連続で改善しなければ停止 過学習防止
            use_best_model=True,  # 最良のiterationまで戻す
        )
        pred_cb = cb.predict(X_va_cb)
        cb_rmses.append(rmse(y_va, pred_cb))
        cb_maes.append(mae(y_va, pred_cb))

        # ★追加：正解率用に保存
        cb_all_true_list.append(y_va.to_numpy())
        cb_all_pred_list.append(np.asarray(pred_cb))

        # ★追加：CatBoostの特徴量重要度（LossFunctionChange系のデフォルト）
        cb_importances.append(cb.get_feature_importance())

        # ---- LightGBM ----
        lgbm = lgb.LGBMRegressor(
            n_estimators=10000,
            learning_rate=0.03,
            num_leaves=64,  # 1本の木の最大葉数
            subsample=0.8,  # 行方向のサンプリング 毎回データの80%を使って木を作る
            colsample_bytree=0.8,  # 列方向サンプリング 毎回80%の特徴量で木を作る
            random_state=run_seed,
            n_jobs=-1,
            verbosity=-1, # ログ非表示
        )
        lgbm.fit(
            X_tr_lgb, y_tr,
            eval_set=[(X_va_lgb, y_va)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
            categorical_feature=cat_cols,  # カテゴリ列を明示
        )
        pred_lgb = lgbm.predict(X_va_lgb, num_iteration=lgbm.best_iteration_)
        lgb_rmses.append(rmse(y_va, pred_lgb))
        lgb_maes.append(mae(y_va, pred_lgb))

        # ★追加：正解率用に保存
        lgb_all_true_list.append(y_va.to_numpy())
        lgb_all_pred_list.append(np.asarray(pred_lgb))

        # ★追加：LightGBMの特徴量重要度（gain）
        lgb_importances.append(lgbm.booster_.feature_importance(importance_type="gain"))

        """
        print(
            f"[Fold {fold}] "
            f"CatBoost RMSE={cb_rmses[-1]:.5f}, MAE={cb_maes[-1]:.5f} | "
            f"LightGBM RMSE={lgb_rmses[-1]:.5f}, MAE={lgb_maes[-1]:.5f}"
        )
        """

    # ★追加：全foldをまとめて正解率計算
    cb_all_true = np.concatenate(cb_all_true_list)
    cb_all_pred = np.concatenate(cb_all_pred_list)
    lgb_all_true = np.concatenate(lgb_all_true_list)
    lgb_all_pred = np.concatenate(lgb_all_pred_list)

    cb_acc = tolerance_accuracy(cb_all_true, cb_all_pred, tolerance_ratio=0.05)
    lgb_acc = tolerance_accuracy(lgb_all_true, lgb_all_pred, tolerance_ratio=0.05)

    def summarize(name: str, rmses: list[float], maes: list[float], acc: float) -> None:
        # print(f"\n== {name} ==")
        # print(f"RMSE mean={np.mean(rmses):.5f}, std={np.std(rmses):.5f}")
        # print(f"MAE  mean={np.mean(maes):.5f}, std={np.std(maes):.5f}")
        # print(f"{name} {np.mean(rmses):.5f} {np.std(rmses):.5f} {np.mean(maes):.5f} {np.std(maes):.5f}",end="")
        # print(f"{name} {np.mean(rmses):.5f} {np.mean(maes):.5f}",end="")
        print(f"{name} {np.mean(rmses):.5f} {np.mean(maes):.5f} {acc:.5f}", end="")

    timestamp = datetime.now().strftime("%m%d%H%M%S")
    timestamp_console=datetime.now().strftime("%m/%d %H:%M:%S")

    summarize(timestamp_console, cb_rmses, cb_maes, cb_acc)
    print(" ", end="")
    summarize("", lgb_rmses, lgb_maes, lgb_acc)
    print()

    # print("\n== Difference (LightGBM - CatBoost) ==")
    # print(f"RMSE diff mean={np.mean(np.array(lgb_rmses) - np.array(cb_rmses)):.5f}")
    # print(f"MAE  diff mean={np.mean(np.array(lgb_maes) - np.array(cb_maes)):.5f}")

    # ------------------------
    # 5) 全フォールド平均の寄与率(%)を出力
    # ------------------------
    # print("\n===== Feature Importance (Mean over folds) =====")

    # CatBoost
    mean_cb_imp = np.mean(np.asarray(cb_importances), axis=0)
    cb_imp_df = to_contrib_df(list(X_cb.columns), mean_cb_imp)

    # print("\n--- CatBoost (Mean Importance + Contribution%) ---")
    # print(cb_imp_df.head(args.top_k))

    # LightGBM
    mean_lgb_imp = np.mean(np.asarray(lgb_importances), axis=0)
    lgb_imp_df = to_contrib_df(list(X_lgb.columns), mean_lgb_imp)

    # print("\n--- LightGBM (Mean Importance[gain] + Contribution%) ---")
    # print(lgb_imp_df.head(args.top_k))

    # ------------------------
    # Excel保存（シート追加）
    # ------------------------


    # ---- CatBoost ----
    cat_file = "result/result_catboost_"+args.result+"_100.xlsx"

    if Path(cat_file).exists():
        with pd.ExcelWriter(cat_file, engine="openpyxl", mode="a", if_sheet_exists="new") as writer:
            cb_imp_df.to_excel(writer, sheet_name=timestamp, index=False)
    else:
        with pd.ExcelWriter(cat_file, engine="openpyxl") as writer:
            cb_imp_df.to_excel(writer, sheet_name=timestamp, index=False)

    # ---- LightGBM ----
    lgb_file = "result/result_lightgbm_"+args.result+"_100.xlsx"

    if Path(lgb_file).exists():
        with pd.ExcelWriter(lgb_file, engine="openpyxl", mode="a", if_sheet_exists="new") as writer:
            lgb_imp_df.to_excel(writer, sheet_name=timestamp, index=False)
    else:
        with pd.ExcelWriter(lgb_file, engine="openpyxl") as writer:
            lgb_imp_df.to_excel(writer, sheet_name=timestamp, index=False)

    # print("\n[Saved to Excel]")
    # print(f"  - {cat_file} (sheet: {timestamp})")
    # print(f"  - {lgb_file} (sheet: {timestamp})")

if __name__ == "__main__":
    timestamp_console=datetime.now().strftime("%m/%d %H:%M:%S")
    print(timestamp_console+" start")
    print()
    print("     timestamp  CB_RMSE   CB_MAE  CB_ACC   LG_RMSE   LG_MAE  LG_ACC")
    for count in range(100):
        main(run_seed=42 + count)