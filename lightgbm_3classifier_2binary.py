import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import train_test_split


def predict_ordinal_binary_with_thresholds(model_ge1, model_ge2, X, t_ge1, t_ge2):
    """
    しきい値を使って 0, 1, 2 を予測する。

    t_ge1:
        P(y>=1) がこの値未満なら 0 と予測

    t_ge2:
        P(y>=2) がこの値以上なら 2 と予測
    """
    p_ge1 = model_ge1.predict_proba(X)[:, 1]
    p_ge2 = model_ge2.predict_proba(X)[:, 1]

    # 順序制約: P(y>=2) <= P(y>=1)
    p_ge2 = np.minimum(p_ge2, p_ge1)

    pred = np.ones(len(X), dtype=int)

    pred[p_ge1 < t_ge1] = 0
    pred[p_ge2 >= t_ge2] = 2

    return pred


def evaluate_metric(y_true, y_pred, metric_name):
    if metric_name == "macro_f1":
        return f1_score(y_true, y_pred, average="macro")
    elif metric_name == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    elif metric_name == "qwk":
        return cohen_kappa_score(y_true, y_pred, weights="quadratic")
    elif metric_name == "negative_mae":
        return -mean_absolute_error(y_true, y_pred)
    else:
        raise ValueError(f"未対応のmetric_nameです: {metric_name}")


def search_best_thresholds(
    model_ge1,
    model_ge2,
    X_valid,
    y_valid,
    metric_name="macro_f1",
):
    """
    検証データ上で t_ge1, t_ge2 を探索する。
    外側のtest foldは使わないので、評価リークを避けられる。
    """
    best_score = -np.inf
    best_t_ge1 = 0.5
    best_t_ge2 = 0.5

    # 0と2を拾いやすくするため、広めに探索
    t_ge1_candidates = np.arange(0.30, 0.71, 0.02)
    t_ge2_candidates = np.arange(0.30, 0.71, 0.02)

    for t_ge1 in t_ge1_candidates:
        for t_ge2 in t_ge2_candidates:
            pred = predict_ordinal_binary_with_thresholds(
                model_ge1=model_ge1,
                model_ge2=model_ge2,
                X=X_valid,
                t_ge1=t_ge1,
                t_ge2=t_ge2,
            )

            score = evaluate_metric(y_valid, pred, metric_name)

            if score > best_score:
                best_score = score
                best_t_ge1 = t_ge1
                best_t_ge2 = t_ge2

    return best_t_ge1, best_t_ge2, best_score

def make_unique_names(names):
    seen = {}
    unique = []

    for name in names:
        base = str(name).strip()
        if base == "" or base.lower() == "nan":
            base = "col"

        if base not in seen:
            seen[base] = 1
            unique.append(base)
        else:
            seen[base] += 1
            unique.append(f"{base}_{seen[base]}")

    return unique


def load_excel_data(path, sheet_name=0):
    """
    Excel形式:
      0行目: 列タイトル
      1行目: 列ID
      2行目以降: データ

    最終列以外: 説明変数
    最終列: 目的変数 0, 1, 2
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")

    if raw.shape[0] < 3:
        raise ValueError("Excelには少なくとも3行必要です: 列タイトル、列ID、データ行。")

    titles = raw.iloc[0].tolist()
    ids = raw.iloc[1].tolist()

    col_names = []
    for i, (title, col_id) in enumerate(zip(titles, ids)):
        if pd.notna(col_id) and str(col_id).strip() != "":
            col_names.append(str(col_id).strip())
        elif pd.notna(title) and str(title).strip() != "":
            col_names.append(str(title).strip())
        else:
            col_names.append(f"col_{i + 1:03d}")

    col_names = make_unique_names(col_names)

    df = raw.iloc[2:].copy()
    df.columns = col_names
    df = df.dropna(how="all")

    if df.shape[1] < 2:
        raise ValueError("説明変数と目的変数を含めて、少なくとも2列必要です。")

    feature_cols = df.columns[:-1].tolist()
    target_col = df.columns[-1]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    y_num = pd.to_numeric(df[target_col], errors="coerce")
    valid_y = y_num.notna()

    if valid_y.sum() < len(y_num):
        print(f"[警告] 目的変数が欠損または数値変換不可の行を {len(y_num) - valid_y.sum()} 行除外しました。")

    X = X.loc[valid_y].reset_index(drop=True)
    y_num = y_num.loc[valid_y].reset_index(drop=True)

    if not np.allclose(y_num, np.round(y_num)):
        raise ValueError("目的変数には 0, 1, 2 の整数ラベルを指定してください。")

    y = y_num.round().astype(int)

    labels = sorted(y.unique().tolist())
    if labels != [0, 1, 2]:
        raise ValueError(f"目的変数のクラスは [0, 1, 2] である必要があります。現在のクラス: {labels}")

    title_map = {}
    for name, title in zip(col_names, titles):
        title_map[name] = "" if pd.isna(title) else str(title)

    return X, y, feature_cols, target_col, title_map


def build_binary_model(seed):
    """
    y>=1, y>=2 の二値分類用LightGBM。
    データ数1583程度を想定し、やや保守的な設定にしている。
    """
    return LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.02,
        num_leaves=7,
        max_depth=3,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        importance_type="gain",
    )


def get_balanced_sample_weight(y_binary):
    """
    二値分類用の簡易balanced sample weight。
    クラス0とクラス1の合計重みが同じくらいになるようにする。

    注意:
    重みを使うと確率の校正がやや崩れる場合があるため、
    デフォルトでは使わず、--balanced指定時のみ使う。
    """
    y_binary = np.asarray(y_binary)
    n = len(y_binary)
    counts = np.bincount(y_binary, minlength=2)

    weights = np.ones(n, dtype=float)

    for cls in [0, 1]:
        if counts[cls] > 0:
            weights[y_binary == cls] = n / (2.0 * counts[cls])

    return weights


def fit_ordinal_binary_models(X_train, y_train, seed, use_balanced_weight=False):
    """
    2つの二値分類器を学習する。

    model_ge1: y >= 1
    model_ge2: y >= 2
    """
    y_ge1 = (y_train >= 1).astype(int)
    y_ge2 = (y_train >= 2).astype(int)

    model_ge1 = build_binary_model(seed=seed)
    model_ge2 = build_binary_model(seed=seed + 100000)

    if use_balanced_weight:
        w_ge1 = get_balanced_sample_weight(y_ge1)
        w_ge2 = get_balanced_sample_weight(y_ge2)

        model_ge1.fit(X_train, y_ge1, sample_weight=w_ge1)
        model_ge2.fit(X_train, y_ge2, sample_weight=w_ge2)
    else:
        model_ge1.fit(X_train, y_ge1)
        model_ge2.fit(X_train, y_ge2)

    return model_ge1, model_ge2


def predict_ordinal_binary(model_ge1, model_ge2, X):
    """
    2つの二値分類器から、最終的な 0, 1, 2 の予測を作る。

    p_ge1 = P(y >= 1)
    p_ge2 = P(y >= 2)

    独立に学習しているため、まれに p_ge2 > p_ge1 になる。
    その場合は p_ge2 を p_ge1 以下に補正する。
    """
    p_ge1 = model_ge1.predict_proba(X)[:, 1]
    p_ge2 = model_ge2.predict_proba(X)[:, 1]

    # 順序制約の補正: P(y>=2) <= P(y>=1)
    p_ge2 = np.minimum(p_ge2, p_ge1)

    p0 = 1.0 - p_ge1
    p1 = p_ge1 - p_ge2
    p2 = p_ge2

    proba = np.vstack([p0, p1, p2]).T

    # 数値誤差対策
    proba = np.clip(proba, 0.0, 1.0)
    proba_sum = proba.sum(axis=1, keepdims=True)
    proba = proba / proba_sum

    pred = np.argmax(proba, axis=1)

    return pred, proba


def main():
    parser = argparse.ArgumentParser(
        description="Excelデータを用いたLightGBM順序付き3クラス分類: 2つの二値分類器版"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="入力Excelファイルのパス。例: data.xlsx",
    )

    parser.add_argument(
        "--sheet",
        default=0,
        help="Excelシート名または番号。デフォルトは0番目のシート。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード。デフォルト: 42",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="k分割交差検証の分割数。デフォルト: 5",
    )

    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="交差検証の繰り返し回数。デフォルト: 10",
    )

    parser.add_argument(
        "--balanced",
        action="store_true",
        help="指定すると、2つの二値分類でbalanced sample weightを使用します。",
    )

    parser.add_argument(
        "--model-out",
        default=None,
        help="指定すると、全データで学習した最終モデルを保存します。例: ordinal_lgbm_model.joblib",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {data_path}")

    sheet_name = int(args.sheet) if str(args.sheet).isdigit() else args.sheet

    X, y, feature_cols, target_col, title_map = load_excel_data(
        data_path,
        sheet_name=sheet_name,
    )

    print("========== データ概要 ==========")
    print(f"データ数: {len(X)}")
    print(f"説明変数数: {X.shape[1]}")
    print(f"目的変数列: {target_col}")
    print("クラス分布:")
    print(y.value_counts().sort_index())
    print()

    cv = RepeatedStratifiedKFold(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.seed,
    )

    metrics = []
    all_true = []
    all_pred = []

    importances_ge1 = []
    importances_ge2 = []

    for split_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        repeat_no = (split_idx - 1) // args.n_splits + 1
        fold_no = (split_idx - 1) % args.n_splits + 1

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # 外側CVの訓練データを、さらに学習用・しきい値調整用に分ける
        X_fit, X_valid, y_fit, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=args.seed + split_idx,
        )

        # しきい値探索用モデル
        tmp_model_ge1, tmp_model_ge2 = fit_ordinal_binary_models(
            X_train=X_fit,
            y_train=y_fit,
            seed=args.seed + split_idx,
            use_balanced_weight=args.balanced,
        )

        # しきい値を探索
        best_t_ge1, best_t_ge2, best_threshold_score = search_best_thresholds(
            model_ge1=tmp_model_ge1,
            model_ge2=tmp_model_ge2,
            X_valid=X_valid,
            y_valid=y_valid,
            metric_name="macro_f1",
        )

        # 外側CVの訓練データ全体で最終学習
        model_ge1, model_ge2 = fit_ordinal_binary_models(
            X_train=X_train,
            y_train=y_train,
            seed=args.seed + split_idx,
            use_balanced_weight=args.balanced,
        )

        # 探索したしきい値でtest foldを予測
        pred = predict_ordinal_binary_with_thresholds(
            model_ge1=model_ge1,
            model_ge2=model_ge2,
            X=X_test,
            t_ge1=best_t_ge1,
            t_ge2=best_t_ge2,
        )

        row = {
            "repeat": repeat_no,
            "fold": fold_no,
            "accuracy": accuracy_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro"),
            "quadratic_weighted_kappa": cohen_kappa_score(
                y_test,
                pred,
                weights="quadratic",
            ),
            "label_mae": mean_absolute_error(y_test, pred),
        }

        metrics.append(row)
        all_true.extend(y_test.tolist())
        all_pred.extend(pred.tolist())

        importances_ge1.append(model_ge1.feature_importances_)
        importances_ge2.append(model_ge2.feature_importances_)

        print(
            f"repeat={repeat_no:02d}, fold={fold_no:02d} | "
            f"macro_f1={row['macro_f1']:.4f}, "
            f"balanced_acc={row['balanced_accuracy']:.4f}, "
            f"qwk={row['quadratic_weighted_kappa']:.4f}, "
            f"mae={row['label_mae']:.4f}, "
            f"t_ge1={best_t_ge1:.2f}, "
            f"t_ge2={best_t_ge2:.2f}"
        )

    metrics_df = pd.DataFrame(metrics)

    print()
    print("========== 交差検証結果: 平均 ==========")
    print(metrics_df.drop(columns=["repeat", "fold"]).mean().round(4))

    print()
    print("========== 交差検証結果: 標準偏差 ==========")
    print(metrics_df.drop(columns=["repeat", "fold"]).std().round(4))

    print()
    print("========== 混同行列 ==========")
    print("行: 正解, 列: 予測")
    print(confusion_matrix(all_true, all_pred, labels=[0, 1, 2]))

    print()
    print("========== 分類レポート ==========")
    print(
        classification_report(
            all_true,
            all_pred,
            labels=[0, 1, 2],
            digits=4,
        )
    )

    mean_importance_ge1 = np.mean(np.vstack(importances_ge1), axis=0)
    mean_importance_ge2 = np.mean(np.vstack(importances_ge2), axis=0)

    importance_df = pd.DataFrame(
        {
            "feature_id": feature_cols,
            "feature_title": [title_map.get(c, "") for c in feature_cols],
            "importance_y_ge_1": mean_importance_ge1,
            "importance_y_ge_2": mean_importance_ge2,
        }
    )

    importance_df["importance_mean"] = (
        importance_df["importance_y_ge_1"] + importance_df["importance_y_ge_2"]
    ) / 2.0

    importance_df = importance_df.sort_values("importance_mean", ascending=False)

    print()
    print("========== 特徴量重要度 Top 30 ==========")
    print(importance_df.head(30).to_string(index=False))

    if args.model_out is not None:
        final_model_ge1, final_model_ge2 = fit_ordinal_binary_models(
            X_train=X,
            y_train=y,
            seed=args.seed,
            use_balanced_weight=args.balanced,
        )

        output = {
            "model_ge1": final_model_ge1,
            "model_ge2": final_model_ge2,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "title_map": title_map,
            "description": "Ordinal classification using two binary LightGBM models: y>=1 and y>=2",
        }

        joblib.dump(output, args.model_out)

        print()
        print(f"最終モデルを保存しました: {args.model_out}")


if __name__ == "__main__":
    main()