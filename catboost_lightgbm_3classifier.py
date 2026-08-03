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

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from catboost import CatBoostClassifier
import lightgbm as lgb
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.metrics import classification_report
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
    tolerance_ratio: float = 0.05
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

    # 1件しかないクラスを除外
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    mask = y.isin(valid_classes)
    X_cb = X_cb.loc[mask].reset_index(drop=True)
    X_lgb = X_lgb.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    n_splits = min(5, y.value_counts().min())
    print(f"n_splits: {n_splits}")
    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=run_seed
    )
    cb_importances, lgb_importances = [], []
    all_true_list, cb_all_pred_list, lgb_all_pred_list = [], [], []
    cb_all_proba_list, lgb_all_proba_list = [], []
    all_true = []
    lgb_all_pred = []

    # k分割交差検証開始
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cb, y), start=1):
        X_tr_cb, X_va_cb = X_cb.iloc[tr_idx], X_cb.iloc[va_idx]
        X_tr_lgb, X_va_lgb = X_lgb.iloc[tr_idx], X_lgb.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        all_true_list.append(y_va.to_numpy())

        # CatBoost実行
        """
        cb = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=10000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=10,
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
        pred_cb_proba = cb.predict_proba(X_va_cb)
        pred_label_cb = cb.classes_[np.argmax(pred_cb_proba, axis=1)]
        cb_all_pred_list.append(np.asarray(pred_label_cb))
        cb_all_proba_list.append(np.asarray(pred_cb_proba))
        cb_importances.append(cb.get_feature_importance())
        print("cb終了")
        """

        # LightGBM実行
        weights = np.ones(len(y_tr))
        lgbm = lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=10000,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=run_seed,
            n_jobs=-1,
            verbosity=-1,
        )
        print("fold:", fold)
        lgbm.fit(
            X_tr_lgb, y_tr,
            # sample_weight=weights,
            eval_set=[(X_va_lgb, y_va)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
            categorical_feature=cat_cols,
        )
        # 各クラスの予測確率: shape = (n_samples, 3)
        pred_lgb_proba = lgbm.predict_proba(
            X_va_lgb,
            num_iteration=lgbm.best_iteration_
        )

        # 最終的な予測クラス: 0, 1, 2
        pred_lgb = pred_lgb_proba.argmax(axis=1)
        # foldごとの正解ラベルと予測ラベルを保存
        all_true.append(np.asarray(y_va))
        lgb_all_pred.append(np.asarray(pred_lgb))

    all_true = np.concatenate(all_true)
    lgb_all_pred = np.concatenate(lgb_all_pred)

    lgb_acc = accuracy_score(all_true, lgb_all_pred)
    lgb_f1_macro = f1_score(all_true, lgb_all_pred, average="macro")
    lgb_f1_weighted = f1_score(all_true, lgb_all_pred, average="weighted")
    lgb_mae = mean_absolute_error(all_true, lgb_all_pred)

    lgb_acc_pm1 = np.mean(np.abs(all_true - lgb_all_pred) <= 1)
    lgb_acc_pm2 = np.mean(np.abs(all_true - lgb_all_pred) <= 2)

    result = {
        "run_no": run_no,

        "LGB_ACC": lgb_acc,
        "LGB_F1_MACRO": lgb_f1_macro,
        "LGB_F1_WEIGHTED": lgb_f1_weighted,
        "LGB_MAE": lgb_mae,
        # "LGB_RMSE": lgb_rmse,
        "LGB_ACC_PM1": lgb_acc_pm1,
        "LGB_ACC_PM2": lgb_acc_pm2,
    }
    # =========================
    # 混同行列
    # =========================

    labels = sorted(np.unique(all_true))

    cm = confusion_matrix(
        all_true,
        lgb_all_pred,
        labels=labels
    )
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    )
    print(cm_df)

    return result


def main():
    filename = "result/result_" + datetime.now().strftime("%Y%m%dT%H%M%S")+".xlsx"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="データのExcelファイルパス")
    parser.add_argument("--result", default=filename, help="結果保存用Excelファイルパス")
    parser.add_argument("--n_splits", type=int, default=5, help="k分割交差検証のkの値")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--ratio", type=float, default=0.05, help="正解率")
    parser.add_argument("--num_exec", type=int, default=10, help="k分割交差検証の回数")
    args = parser.parse_args()
    data_file = args.data
    result_file = args.result

    timestamp = datetime.now().strftime("%m/%d %H:%M:%S")
    print(timestamp+" start")
    print()
    # print("     timestamp   no CB_RMSE CB_MAE CB_ACC LG_RMSE LG_MAE LG_ACC")

    result_list = []
    cb_imp_df_list = []
    lgb_imp_df_list = []

    # k分割交差検証を実行
    for count in range(args.num_exec):
        # result, cb_imp_df, lgb_imp_df = run_experiment(
        result = run_experiment(
            run_seed=args.seed + count,
            run_no=count + 1,
            data_file=data_file,
            n_splits=args.n_splits,
            tolerance_ratio=args.ratio,
        )
        timestamp = datetime.now().strftime("%m/%d %H:%M:%S")
        print(f"run_no              : {result['run_no']}")

        print(f"LGB_ACC             : {result['LGB_ACC']:.6f}")
        print(f"LGB_F1_MACRO        : {result['LGB_F1_MACRO']:.6f}")
        print(f"LGB_F1_WEIGHTED     : {result['LGB_F1_WEIGHTED']:.6f}")
        print(f"LGB_MAE             : {result['LGB_MAE']:.6f}")
        # print(f"LGB_RMSE            : {result['LGB_RMSE']:.6f}")
        print(f"LGB_ACC_PM1         : {result['LGB_ACC_PM1']:.6f}")
        print(f"LGB_ACC_PM2         : {result['LGB_ACC_PM2']:.6f}")

    return

    result_df = pd.DataFrame(result_list)
    cb_pivot_df = build_importance_pivot_df(cb_imp_df_list)
    lgb_pivot_df = build_importance_pivot_df(lgb_imp_df_list)

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


if __name__ == "__main__":
    main()
