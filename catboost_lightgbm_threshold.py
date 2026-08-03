# pip install pandas openpyxl scikit-learn catboost lightgbm
# python catboost_lightgbm.py --data data_204項目_male_空腹時.xlsx
#
# 「--(項目名) (値)」を追加することで、以下の値を指定できます。
# --data: データのExcelファイルパス (必須)
# --result: 結果保存用Excelファイルパス (任意 デフォルトは日付と時刻)
# --n_splits: k分割交差検証のkの値 (任意 デフォルトは5)
# --seed: 乱数シード (任意: デフォルトは42)
# --ratio: 正解率 (任意: デフォルトは0.05)
# --num_exec: k分割交差検証の回数 (任意: デフォルトは10)
# --threshold1: 目的変数を3分類する際の1つ目のしきい値 (必須)
# --threshold2: 目的変数を3分類する際の2つ目のしきい値 (必須)

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from catboost import CatBoostRegressor
import lightgbm as lgb
from datetime import datetime


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


def classify_three_classes(values, threshold1: float, threshold2: float) -> np.ndarray:
    # class_0: value < threshold1
    # class_1: threshold1 <= value < threshold2
    # class_2: threshold2 <= value
    values = np.asarray(values, dtype=float)
    return np.digitize(values, bins=[threshold1, threshold2], right=False)


def class_match_accuracy(
    y_true,
    y_pred,
    threshold1: float,
    threshold2: float,
) -> float:
    true_class = classify_three_classes(y_true, threshold1, threshold2)
    pred_class = classify_three_classes(y_pred, threshold1, threshold2)
    return float(np.mean(true_class == pred_class))


def build_confusion_matrix_values(
    y_true,
    y_pred,
    threshold1: float,
    threshold2: float,
    prefix: str,
):
    true_class = classify_three_classes(y_true, threshold1, threshold2)
    pred_class = classify_three_classes(y_pred, threshold1, threshold2)
    matrix = confusion_matrix(true_class, pred_class, labels=[0, 1, 2])

    values = {
        f"{prefix}_class_match_accuracy": float(np.mean(true_class == pred_class)),
    }
    for true_label in range(3):
        for pred_label in range(3):
            values[f"{prefix}_true_{true_label}_pred_{pred_label}"] = int(
                matrix[true_label, pred_label]
            )
    return values


# 重要度のDataFrameを生成
def build_importance_df(
    features: list[str],
    importances_list: list[np.ndarray],
    item_map_df: pd.DataFrame,
    run_no: int,
):
    mean_importance = np.mean(
        np.asarray(importances_list, dtype=float),
        axis=0,
    )
    total = float(np.sum(mean_importance))
    contribution = mean_importance / total * 100.0
    imp_df = pd.DataFrame({
        "run_no": run_no,
        "Feature": pd.Series(features, dtype="str"),
        "Contribution": contribution,
    })
    imp_df = imp_df.merge(
        item_map_df,
        on="Feature",
        how="left",
    )
    imp_df = imp_df[
        ["run_no", "Feature", "name", "Contribution"]
    ]
    return imp_df


# 重要度のDataFrameを結合
def build_importance_pivot_df(imp_df_list):
    all_imp_df = pd.concat(
        imp_df_list,
        axis=0,
        ignore_index=True,
    )
    pivot_df = all_imp_df.pivot(
        index=["Feature", "name"],
        columns="run_no",
        values="Contribution",
    )
    pivot_df.columns = [
        f"run_{col}" for col in pivot_df.columns
    ]
    pivot_df = pivot_df.reset_index()
    pivot_df["Feature"] = pd.to_numeric(pivot_df["Feature"])
    return pivot_df


# 予測を実行
def run_experiment(
    run_seed: int = 42,
    run_no: int = 1,
    data_file: str = "",
    n_splits: int = 5,
    tolerance_ratio: float = 0.05,
    threshold1: float = 0.0,
    threshold2: float = 0.0,
):
    raw = pd.read_excel(data_file, header=None, engine="openpyxl")

    # 1行目の項目名、2行目の項目IDを取得
    item_names = raw.iloc[0, :].astype(str).tolist()
    item_ids = raw.iloc[1, :].tolist()
    item_map_df = pd.DataFrame({
        "Feature": item_ids,
        "name": item_names,
    })
    item_map_df["Feature"] = item_map_df["Feature"].astype(str)
    col_ids = [str(x) for x in item_ids]

    # 3行目以降のデータを取得
    df = raw.iloc[2:, :].copy()
    df.columns = col_ids
    df.reset_index(drop=True, inplace=True)

    # 最終列の目的変数を取得
    target_col = df.columns[-1]
    feature_cols = list(df.columns[:-1])

    # 項目が数値かカテゴリかを判定
    name_by_id = dict(zip(col_ids, item_names))
    cat_cols = [cid for cid in feature_cols if name_by_id.get(
        cid, "").strip().lower().startswith("c")]
    num_cols = [cid for cid in feature_cols if name_by_id.get(
        cid, "").strip().lower().startswith("n")]
    other_cols = [
        cid for cid in feature_cols if cid not in cat_cols and cid not in num_cols]
    if other_cols:
        print("項目名が c/n で始まらない列があります。")
        num_cols += other_cols

    # 説明変数と目的変数を指定
    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target_col], errors="coerce")
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # カテゴリ列を文字列に変換
    X_cb = X.copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].where(~X_cb[c].isna(), "missing")
        X_cb[c] = X_cb[c].astype(str)
    cat_feature_indices = [X_cb.columns.get_loc(c) for c in cat_cols]
    X_lgb = X.copy()
    for c in cat_cols:
        X_lgb[c] = X_lgb[c].where(~X_lgb[c].isna(), "missing").astype(
            str).astype("category")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=run_seed)
    cb_importances, lgb_importances = [], []
    all_true_list, cb_all_pred_list, lgb_all_pred_list = [], [], []
    matrix_rows = []

    # k分割交差検証開始
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb), start=1):
        X_tr_cb, X_va_cb = X_cb.iloc[tr_idx], X_cb.iloc[va_idx]
        X_tr_lgb, X_va_lgb = X_lgb.iloc[tr_idx], X_lgb.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        all_true_list.append(y_va.to_numpy())

        # CatBoost実行
        cb = CatBoostRegressor(
            loss_function="RMSE",  # 学習時に最小化する目的関数
            eval_metric="RMSE",  # 検証データで見る指標
            iterations=10000,  # 最大ツリー数(ブースティング回数)
            learning_rate=0.03,  # 1本の木がどれだけ強く補正するか
            depth=6,  # 1本の木の深さの最大(深いほど過学習しやすい)
            l2_leaf_reg=10,  # L2正則化(過学習抑制)
            random_seed=run_seed,
            allow_writing_files=False,
            verbose=False,
        )
        cb.fit(
            X_tr_cb, y_tr,
            cat_features=cat_feature_indices,
            eval_set=(X_va_cb, y_va),
            early_stopping_rounds=200,  # この回数連続で改善しなければ停止
            use_best_model=True,
        )
        pred_cb = cb.predict(X_va_cb)
        pred_cb = np.asarray(pred_cb)
        cb_all_pred_list.append(pred_cb)
        cb_importances.append(cb.get_feature_importance())

        # LightGBM実行
        lgbm = lgb.LGBMRegressor(
            n_estimators=10000,
            learning_rate=0.03,
            num_leaves=64,  # 1本の木の最大葉数
            subsample=0.8,  # 行方向のサンプリング(データの何%を使って木を作るか)
            colsample_bytree=0.8,  # 列方向サンプリング(何%の特徴量で木を作るか)
            random_state=run_seed,
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
        pred_lgb = np.asarray(pred_lgb)
        lgb_all_pred_list.append(pred_lgb)
        lgb_importances.append(
            lgbm.booster_.feature_importance(importance_type="gain"))
        matrix_row = {
            "run_no": run_no,
            "n_splits": n_splits,
            "fold": fold,
            "threshold1": threshold1,
            "threshold2": threshold2,
        }
        matrix_row.update(build_confusion_matrix_values(
            y_true=y_va,
            y_pred=pred_cb,
            threshold1=threshold1,
            threshold2=threshold2,
            prefix="CB",
        ))
        matrix_row.update(build_confusion_matrix_values(
            y_true=y_va,
            y_pred=pred_lgb,
            threshold1=threshold1,
            threshold2=threshold2,
            prefix="LG",
        ))
        matrix_rows.append(matrix_row)

    all_true = np.concatenate(all_true_list)
    cb_all_pred = np.concatenate(cb_all_pred_list)
    lgb_all_pred = np.concatenate(lgb_all_pred_list)

    # MAE、RMSE
    cb_mae_all = mae(all_true, cb_all_pred)
    lgb_mae_all = mae(all_true, lgb_all_pred)
    cb_rmse_all = rmse(all_true, cb_all_pred)
    lgb_rmse_all = rmse(all_true, lgb_all_pred)

    # 正解率
    cb_acc = tolerance_accuracy(all_true, cb_all_pred, tolerance_ratio)
    lgb_acc = tolerance_accuracy(all_true, lgb_all_pred, tolerance_ratio)

    # 3分類した真値と予測値のクラス一致率
    cb_class_acc = class_match_accuracy(
        all_true, cb_all_pred, threshold1, threshold2
    )
    lgb_class_acc = class_match_accuracy(
        all_true, lgb_all_pred, threshold1, threshold2
    )

    result = {
        "run_no": run_no,
        "threshold1": threshold1,
        "threshold2": threshold2,
        "CB_RMSE": cb_rmse_all,
        "CB_MAE": cb_mae_all,
        "CB_ACC": cb_acc,
        "CB_CLASS_ACC": cb_class_acc,
        "LG_RMSE": lgb_rmse_all,
        "LG_MAE": lgb_mae_all,
        "LG_ACC": lgb_acc,
        "LG_CLASS_ACC": lgb_class_acc,
    }

    # 重要度のDataFrameを生成
    cb_imp_df = build_importance_df(
        features=list(X_cb.columns),
        importances_list=cb_importances,
        item_map_df=item_map_df,
        run_no=run_no,
    )
    lgb_imp_df = build_importance_df(
        features=list(X_lgb.columns),
        importances_list=lgb_importances,
        item_map_df=item_map_df,
        run_no=run_no,
    )
    matrix_df = pd.DataFrame(matrix_rows)
    return result, cb_imp_df, lgb_imp_df, matrix_df


def main():
    filename = "result/result_" + datetime.now().strftime("%Y%m%dT%H%M%S")+".xlsx"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="データのExcelファイルパス")
    parser.add_argument("--result", default=filename, help="結果保存用Excelファイルパス")
    parser.add_argument("--n_splits", type=int, default=5, help="k分割交差検証のkの値")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--ratio", type=float, default=0.05, help="正解率")
    parser.add_argument("--num_exec", type=int, default=10, help="k分割交差検証の回数")
    parser.add_argument(
        "--threshold1",
        type=float,
        required=True,
        help="目的変数を3分類する際の1つ目のしきい値",
    )
    parser.add_argument(
        "--threshold2",
        type=float,
        required=True,
        help="目的変数を3分類する際の2つ目のしきい値",
    )
    args = parser.parse_args()
    if args.threshold1 >= args.threshold2:
        parser.error("--threshold1 には --threshold2 より小さい値を指定してください。")
    data_file = args.data
    result_file = args.result

    timestamp = datetime.now().strftime("%m/%d %H:%M:%S")
    print(timestamp+" start")
    print()
    print("     timestamp   no CB_RMSE CB_MAE CB_ACC CB_CLASS_ACC LG_RMSE LG_MAE LG_ACC LG_CLASS_ACC")

    result_list = []
    cb_imp_df_list = []
    lgb_imp_df_list = []
    matrix_df_list = []

    # k分割交差検証を実行
    for count in range(args.num_exec):
        result, cb_imp_df, lgb_imp_df, matrix_df = run_experiment(
            run_seed=args.seed + count,
            run_no=count + 1,
            data_file=data_file,
            n_splits=args.n_splits,
            tolerance_ratio=args.ratio,
            threshold1=args.threshold1,
            threshold2=args.threshold2,
        )
        timestamp = datetime.now().strftime("%m/%d %H:%M:%S")
        print(
            f"{timestamp} {result['run_no']:4d}   "
            f"{result['CB_RMSE']:.2f}  "
            f"{result['CB_MAE']:.2f} "
            f"{result['CB_ACC']:.4f} "
            f"{result['CB_CLASS_ACC']:.4f} "
            f"  {result['LG_RMSE']:.2f}  "
            f"{result['LG_MAE']:.2f} "
            f"{result['LG_ACC']:.4f} "
            f"{result['LG_CLASS_ACC']:.4f}"
        )
        result_list.append(result)
        cb_imp_df_list.append(cb_imp_df)
        lgb_imp_df_list.append(lgb_imp_df)
        matrix_df_list.append(matrix_df)

    result_df = pd.DataFrame(result_list)
    cb_pivot_df = build_importance_pivot_df(cb_imp_df_list)
    lgb_pivot_df = build_importance_pivot_df(lgb_imp_df_list)
    matrix_df = pd.concat(matrix_df_list, axis=0, ignore_index=True)

    with pd.ExcelWriter(result_file, engine="openpyxl") as writer:
        result_df.to_excel(writer, index=False, sheet_name="result")
        cb_pivot_df.to_excel(
            writer,
            sheet_name="catboost_importance",
            index=False,
        )
        lgb_pivot_df.to_excel(
            writer,
            sheet_name="lightgbm_importance",
            index=False,
        )
        matrix_df.to_excel(
            writer,
            sheet_name="matrix",
            index=False,
        )


if __name__ == "__main__":
    main()
