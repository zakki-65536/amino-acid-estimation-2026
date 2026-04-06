# predict_catboost_lightgbm_10runs_to_excel.py
# ------------------------------------------------------------
# Excel形式（あなたの形式）:
# - 1行目: 項目名（先頭 c=カテゴリ, n=数値）
# - 2行目: 項目ID（学習時の列名として使用）
# - 3行目以降: データ
# - 最終列が目的変数（数値）
#
# 目的:
# - CatBoost と LightGBM を「同じ train/test 分割」で 10回実行（分割は毎回ランダム）
# - 各回のテストデータに対する予測値を「真値(y_true)と並べて」Excel出力
#
# 出力:
# - 1つのExcel（--out で指定、既定: result_predictions.xlsx）
# - シートは Run01..Run10（各回のテスト行のみ）
# - Summary シートに各回のRMSE/MAEと設定を記録
#
# 必要:
#   pip install pandas openpyxl scikit-learn catboost lightgbm
# ------------------------------------------------------------

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from catboost import CatBoostRegressor
import lightgbm as lgb


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def make_unique_names(names: list[str]) -> list[str]:
    """重複列名があると学習で事故るので、重複時は _2, _3... を付与してユニーク化"""
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        n = str(n)
        if n not in seen:
            seen[n] = 1
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
    return out


@dataclass
class DataPack:
    X_cb: pd.DataFrame
    X_lgb: pd.DataFrame
    y: pd.Series
    feature_cols: list[str]
    cat_cols: list[str]
    cat_feature_indices: list[int]
    target_col: str
    # 元の行番号（Excel上は3行目開始なので、Excel行を復元したいときに使える）
    source_row_index: pd.Series  # 0-based in df after dropping header rows


def load_and_prepare(excel_path: str, sheet=0) -> DataPack:
    raw = pd.read_excel(excel_path, sheet_name=sheet, header=None, engine="openpyxl")

    if raw.shape[0] < 3:
        raise ValueError("行数が足りません（最低でも1行目=項目名, 2行目=項目ID, 3行目以降=データが必要）")
    if raw.shape[1] < 2:
        raise ValueError("列数が足りません（説明変数＋目的変数が必要）")

    item_names = raw.iloc[0, :].astype(str).tolist()
    item_ids = raw.iloc[1, :].tolist()

    col_ids = make_unique_names([str(x) for x in item_ids])

    df = raw.iloc[2:, :].copy()
    df.columns = col_ids
    df.reset_index(drop=True, inplace=True)

    target_col = df.columns[-1]
    feature_cols = list(df.columns[:-1])

    name_by_id = dict(zip(col_ids, item_names))
    cat_cols = [cid for cid in feature_cols if name_by_id.get(cid, "").strip().lower().startswith("c")]
    num_cols = [cid for cid in feature_cols if name_by_id.get(cid, "").strip().lower().startswith("n")]

    other_cols = [cid for cid in feature_cols if cid not in cat_cols and cid not in num_cols]
    if other_cols:
        print(f"[WARN] 項目名が c/n で始まらない列が {len(other_cols)} 個あります。数値扱いに寄せます（ID）: {other_cols[:10]}")
        num_cols += other_cols

    # 目的変数を数値化
    y = pd.to_numeric(df[target_col], errors="coerce")

    # 目的変数NaNは除外（学習不能）
    valid_idx = y.notna()
    if valid_idx.sum() != len(y):
        print(f"[WARN] 目的変数がNaNの行を除外します: {len(y) - valid_idx.sum()} 行")
        df = df.loc[valid_idx].reset_index(drop=True)
        y = y.loc[valid_idx].reset_index(drop=True)

    source_row_index = pd.Series(df.index, name="row_index")

    # 説明変数
    X = df[feature_cols].copy()

    # 数値列を数値化（変換不可はNaN）
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # CatBoost用：カテゴリ列は文字列（欠損は "missing"）
    X_cb = X.copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].where(~X_cb[c].isna(), "missing").astype(str)

    # LightGBM用：カテゴリ列は category dtype（欠損は "missing"）
    X_lgb = X.copy()
    for c in cat_cols:
        X_lgb[c] = X_lgb[c].where(~X_lgb[c].isna(), "missing").astype(str).astype("category")

    cat_feature_indices = [X_cb.columns.get_loc(c) for c in cat_cols]

    return DataPack(
        X_cb=X_cb,
        X_lgb=X_lgb,
        y=y,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        cat_feature_indices=cat_feature_indices,
        target_col=target_col,
        source_row_index=source_row_index,
    )


def fit_predict_once(
    pack: DataPack,
    seed: int,
    test_size: float,
    # CatBoost params
    cb_depth: int,
    cb_lr: float,
    cb_l2: float,
    cb_iterations: int,
    cb_early_stopping: int,
    # LightGBM params
    lgb_num_leaves: int,
    lgb_lr: float,
    lgb_n_estimators: int,
    lgb_early_stopping: int,
) -> tuple[pd.DataFrame, dict]:
    # 同じ分割を両モデルで共有
    idx_all = np.arange(len(pack.y))
    tr_idx, te_idx = train_test_split(idx_all, test_size=test_size, random_state=seed, shuffle=True)

    X_tr_cb, X_te_cb = pack.X_cb.iloc[tr_idx], pack.X_cb.iloc[te_idx]
    X_tr_lgb, X_te_lgb = pack.X_lgb.iloc[tr_idx], pack.X_lgb.iloc[te_idx]
    y_tr, y_te = pack.y.iloc[tr_idx], pack.y.iloc[te_idx]

    # ---- CatBoost ----
    cb = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=cb_iterations,
        learning_rate=cb_lr,
        depth=cb_depth,
        l2_leaf_reg=cb_l2,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
    )
    cb.fit(
        X_tr_cb, y_tr,
        cat_features=pack.cat_feature_indices,
        eval_set=(X_te_cb, y_te),
        early_stopping_rounds=cb_early_stopping,
        use_best_model=True,
    )
    pred_cb = cb.predict(X_te_cb)

    # ---- LightGBM ----
    lgbm = lgb.LGBMRegressor(
        n_estimators=lgb_n_estimators,
        learning_rate=lgb_lr,
        num_leaves=lgb_num_leaves,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    lgbm.fit(
        X_tr_lgb, y_tr,
        eval_set=[(X_te_lgb, y_te)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=lgb_early_stopping, verbose=False)],
        categorical_feature=pack.cat_cols,  # 列名(ID)で明示
    )
    pred_lgb = lgbm.predict(X_te_lgb, num_iteration=lgbm.best_iteration_)

    # 出力DataFrame（テストデータのみ）
    out_df = pd.DataFrame({
        "row_index": pack.source_row_index.iloc[te_idx].to_numpy(),  # 0-based（データ部分の行）
        "y_true": y_te.to_numpy(),
        "pred_catboost": pred_cb,
        "pred_lightgbm": pred_lgb,
    }).sort_values("row_index").reset_index(drop=True)

    # メトリクス
    metrics = {
        "seed": seed,
        "test_size": test_size,
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "CB_RMSE": rmse(y_te, pred_cb),
        "CB_MAE": mae(y_te, pred_cb),
        "LG_RMSE": rmse(y_te, pred_lgb),
        "LG_MAE": mae(y_te, pred_lgb),
        "CB_best_iteration": int(cb.tree_count_),
        "LG_best_iteration": int(getattr(lgbm, "best_iteration_", 0) or 0),
        "cb_depth": cb_depth,
        "cb_lr": cb_lr,
        "cb_l2": cb_l2,
        "lgb_num_leaves": lgb_num_leaves,
        "lgb_lr": lgb_lr,
    }

    return out_df, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True, help="Excelファイルパス (.xlsx)")
    parser.add_argument("--sheet", default=0, help="シート名またはインデックス（既定: 0）")
    parser.add_argument("--out", default="result_predictions.xlsx", help="出力Excel（既定: result_predictions.xlsx）")
    parser.add_argument("--runs", type=int, default=10, help="実行回数（既定: 10）")
    parser.add_argument("--seed", type=int, default=42, help="ベース乱数シード（既定: 42）")
    parser.add_argument("--test_size", type=float, default=0.2, help="テスト割合（既定: 0.2）")

    # HPOで良さそうだった値をデフォルトにしてあります（必要なら変更）
    parser.add_argument("--cb_depth", type=int, default=6)
    parser.add_argument("--cb_lr", type=float, default=0.03)
    parser.add_argument("--cb_l2", type=float, default=10.0)
    parser.add_argument("--cb_iterations", type=int, default=10000)
    parser.add_argument("--cb_early_stopping", type=int, default=200)

    parser.add_argument("--lgb_num_leaves", type=int, default=64)
    parser.add_argument("--lgb_lr", type=float, default=0.03)
    parser.add_argument("--lgb_n_estimators", type=int, default=10000)
    parser.add_argument("--lgb_early_stopping", type=int, default=200)

    args = parser.parse_args()

    pack = load_and_prepare(args.excel, sheet=args.sheet)

    print("---- Data Summary ----")
    print(f"Rows (data): {len(pack.y)}")
    print(f"Features: {len(pack.feature_cols)}")
    print(f"  Categorical: {len(pack.cat_cols)}")
    print(f"Target (last col ID): {pack.target_col}")
    print("----------------------")

    out_path = Path(args.out)
    if out_path.exists():
        # 事故防止：上書き
        out_path.unlink()

    summary_rows: list[dict] = []

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for i in range(1, args.runs + 1):
            run_seed = args.seed + i  # runごとに変える
            df_pred, metrics = fit_predict_once(
                pack=pack,
                seed=run_seed,
                test_size=args.test_size,
                cb_depth=args.cb_depth,
                cb_lr=args.cb_lr,
                cb_l2=args.cb_l2,
                cb_iterations=args.cb_iterations,
                cb_early_stopping=args.cb_early_stopping,
                lgb_num_leaves=args.lgb_num_leaves,
                lgb_lr=args.lgb_lr,
                lgb_n_estimators=args.lgb_n_estimators,
                lgb_early_stopping=args.lgb_early_stopping,
            )

            sheet_name = f"Run{i:02d}"
            df_pred.to_excel(writer, sheet_name=sheet_name, index=False)

            summary_rows.append({"run": i, **metrics})

            print(
                f"[Run {i:02d}] seed={run_seed} "
                f"CB_RMSE={metrics['CB_RMSE']:.5f} CB_MAE={metrics['CB_MAE']:.5f} | "
                f"LG_RMSE={metrics['LG_RMSE']:.5f} LG_MAE={metrics['LG_MAE']:.5f}"
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"\n[Saved] {out_path.resolve()}")


if __name__ == "__main__":
    main()