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


def make_unique_names(names):
    """
    列名が重複しているとLightGBMやpandas処理で問題になるため、重複を避ける。
    """
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

    内部処理では1行目の「列ID」を列名として使う。
    ただし、列IDが空の場合は列タイトルや col_番号 で補完する。
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

    # 列タイトルとの対応表
    title_map = {}
    for name, title in zip(col_names, titles):
        title_map[name] = "" if pd.isna(title) else str(title)

    return X, y, feature_cols, target_col, title_map


def build_model(seed, class_weight=None):
    """
    データ数1583・説明変数30程度を想定し、やや過学習を抑えた初期設定。
    必要に応じて後でチューニングしてください。
    """
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        importance_type="gain",
    )



def export_prediction_excel(
    output_path,
    X,
    y,
    feature_cols,
    target_col,
    title_map,
    predictions_by_repeat,
    importance_by_repeat_df=None,
):
    """
    説明変数、目的変数(真値)、各repeatの予測結果をExcelに出力する。

    出力内容:
      - predictions: 1行=元データ1件。説明変数 + 真値 + 予測結果列。
      - column_titles: 入力Excelの列IDと列タイトルの対応表。
      - feature_importance_by_repeat: 各repeatごとの特徴量重要度(gain)。

    デフォルト設定(--n-repeats 10)では pred_repeat_01 ～ pred_repeat_10 の10列が出力される。
    feature_importance_by_repeat では、各repeat内のfoldの重要度を平均して出力する。
    """
    output_path = Path(output_path)
    if output_path.parent != Path(""):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    true_col = f"{target_col}_true"

    result_df = pd.concat(
        [
            X[feature_cols].reset_index(drop=True),
            y.rename(true_col).reset_index(drop=True),
            predictions_by_repeat.reset_index(drop=True),
        ],
        axis=1,
    )

    # 予測列は欠損がなければ通常の整数列として保存する。
    pred_cols = predictions_by_repeat.columns.tolist()
    for col in pred_cols:
        if result_df[col].isna().any():
            result_df[col] = result_df[col].astype("Int64")
        else:
            result_df[col] = result_df[col].astype(int)

    title_df = pd.DataFrame(
        {
            "column_id": [*feature_cols, target_col],
            "column_role": [*("feature" for _ in feature_cols), "target"],
            "column_title": [title_map.get(c, "") for c in [*feature_cols, target_col]],
        }
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="predictions", index=False)
        title_df.to_excel(writer, sheet_name="column_titles", index=False)
        if importance_by_repeat_df is not None:
            importance_by_repeat_df.to_excel(
                writer,
                sheet_name="feature_importance_by_repeat",
                index=False,
            )

        worksheet = writer.sheets["predictions"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for col_cells in worksheet.columns:
            header = str(col_cells[0].value)
            width = min(max(len(header) + 2, 12), 24)
            worksheet.column_dimensions[col_cells[0].column_letter].width = width

        title_sheet = writer.sheets["column_titles"]
        title_sheet.freeze_panes = "A2"
        title_sheet.auto_filter.ref = title_sheet.dimensions
        for col_cells in title_sheet.columns:
            max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
            title_sheet.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 2, 12), 40)

        if importance_by_repeat_df is not None:
            importance_sheet = writer.sheets["feature_importance_by_repeat"]
            importance_sheet.freeze_panes = "D2"
            importance_sheet.auto_filter.ref = importance_sheet.dimensions
            for col_cells in importance_sheet.columns:
                header = str(col_cells[0].value)
                max_len = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in col_cells
                )
                width = min(max(max_len + 2, len(header) + 2, 12), 32)
                importance_sheet.column_dimensions[col_cells[0].column_letter].width = width

                if header.startswith("importance_gain"):
                    for cell in col_cells[1:]:
                        cell.number_format = "#,##0.000000"
                elif header == "rank_by_mean":
                    for cell in col_cells[1:]:
                        cell.number_format = "0"

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Excelデータを用いたLightGBM 3クラス分類"
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
        help="指定すると class_weight='balanced' を使用します。",
    )

    parser.add_argument(
        "--model-out",
        default=None,
        help="指定すると、全データで学習した最終モデルを保存します。例: model.joblib",
    )

    parser.add_argument(
        "--result",
        default=None,
        help=(
            "指定すると、説明変数・目的変数(真値)・各repeatの予測結果を"
            "Excelに保存します。例: predictions.xlsx"
        ),
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {data_path}")

    # sheet が数値文字列なら int に変換
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

    class_weight = "balanced" if args.balanced else None

    cv = RepeatedStratifiedKFold(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.seed,
    )

    metrics = []
    all_true = []
    all_pred = []
    importances = []
    importances_by_repeat = {repeat_no: [] for repeat_no in range(1, args.n_repeats + 1)}
    predictions_by_repeat = pd.DataFrame(index=X.index)
    for repeat_no in range(1, args.n_repeats + 1):
        predictions_by_repeat[f"pred_repeat_{repeat_no:02d}"] = pd.NA

    for split_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        repeat_no = (split_idx - 1) // args.n_splits + 1
        fold_no = (split_idx - 1) % args.n_splits + 1

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = build_model(
            seed=args.seed + split_idx,
            class_weight=class_weight,
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        pred_col = f"pred_repeat_{repeat_no:02d}"
        predictions_by_repeat.loc[test_idx, pred_col] = pred.astype(int)

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

        fold_importance = np.asarray(model.feature_importances_, dtype=float)
        importances.append(fold_importance)
        importances_by_repeat[repeat_no].append(fold_importance)

        print(
            f"repeat={repeat_no:02d}, fold={fold_no:02d} | "
            f"macro_f1={row['macro_f1']:.4f}, "
            f"balanced_acc={row['balanced_accuracy']:.4f}, "
            f"qwk={row['quadratic_weighted_kappa']:.4f}, "
            f"mae={row['label_mae']:.4f}"
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

    mean_importance = np.mean(np.vstack(importances), axis=0)

    repeat_importance_cols = {}
    for repeat_no in range(1, args.n_repeats + 1):
        repeat_importances = importances_by_repeat.get(repeat_no, [])
        col = f"importance_gain_repeat_{repeat_no:02d}"
        if repeat_importances:
            repeat_importance_cols[col] = np.mean(np.vstack(repeat_importances), axis=0)
        else:
            repeat_importance_cols[col] = np.nan

    importance_by_repeat_df = pd.DataFrame(
        {
            "feature_id": feature_cols,
            "feature_title": [title_map.get(c, "") for c in feature_cols],
            **repeat_importance_cols,
        }
    )
    repeat_importance_col_names = list(repeat_importance_cols.keys())
    importance_by_repeat_df["importance_gain_mean"] = mean_importance
    importance_by_repeat_df = importance_by_repeat_df.sort_values(
        "importance_gain_mean",
        ascending=False,
    ).reset_index(drop=True)
    importance_by_repeat_df.insert(
        2,
        "rank_by_mean",
        np.arange(1, len(importance_by_repeat_df) + 1),
    )

    importance_df = importance_by_repeat_df[
        ["feature_id", "feature_title", "importance_gain_mean"]
    ].copy()

    print()
    print("========== 特徴量重要度 Top 30 ==========")
    print(importance_df.head(30).to_string(index=False))

    if args.result is not None:
        missing_pred_count = int(predictions_by_repeat.isna().sum().sum())
        if missing_pred_count > 0:
            print()
            print(
                f"[警告] 予測結果に未入力セルが {missing_pred_count} 個あります。"
                "n_splits / n_repeats とデータのクラス分布を確認してください。"
            )

        pred_output_path = export_prediction_excel(
            args.result,
            X,
            y,
            feature_cols,
            target_col,
            title_map,
            predictions_by_repeat,
            importance_by_repeat_df=importance_by_repeat_df,
        )
        print()
        print(f"予測結果Excelを保存しました: {pred_output_path}")

    if args.model_out is not None:
        final_model = build_model(
            seed=args.seed,
            class_weight=class_weight,
        )
        final_model.fit(X, y)

        output = {
            "model": final_model,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "title_map": title_map,
        }

        joblib.dump(output, args.model_out)
        print()
        print(f"最終モデルを保存しました: {args.model_out}")


if __name__ == "__main__":
    main()