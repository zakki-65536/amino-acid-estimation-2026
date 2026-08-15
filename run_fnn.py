from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
  accuracy_score,
  balanced_accuracy_score,
  classification_report,
  confusion_matrix,
  f1_score,
  precision_score,
  recall_score,
  roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from filter import filter_data


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"

N_SPLITS = 5
N_REPEATS = 10
RANDOM_STATE = 42

FNN_PARAMS = {
  "hidden_layer_sizes": (64, 32),
  "activation": "relu",
  "solver": "adam",
  "alpha": 0.0001,
  "learning_rate_init": 0.001,
  "max_iter": 1000,
  "early_stopping": True,
  "validation_fraction": 0.1,
  "n_iter_no_change": 30,
  "tol": 0.0001,
}


def create_model(
  split_number: int
) -> Pipeline:
  return Pipeline([
    (
      "imputer",
      SimpleImputer(strategy="median")
    ),
    (
      "scaler",
      StandardScaler()
    ),
    (
      "fnn",
      MLPClassifier(
        **FNN_PARAMS,
        random_state=RANDOM_STATE + split_number
      )
    ),
  ])


def get_repeat_and_fold(
  split_number: int
) -> tuple[int, int]:
  repeat_number = (
    (split_number - 1) // N_SPLITS
  ) + 1

  fold_number = (
    (split_number - 1) % N_SPLITS
  ) + 1

  return repeat_number, fold_number

def calculate_roc_auc(
  y_true: pd.Series,
  probabilities: np.ndarray,
  classes: np.ndarray
) -> float:
  if len(classes) == 2:
    return roc_auc_score(
      y_true,
      probabilities[:, 1]
    )

  return roc_auc_score(
    y_true,
    probabilities,
    labels=classes,
    multi_class="ovr",
    average="macro"
  )


def create_metric_summary(
  fold_metrics: pd.DataFrame
) -> pd.DataFrame:
  metric_names = [
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
  ]

  rows = []

  for metric_name in metric_names:
    values = fold_metrics[metric_name]

    rows.append({
      "metric": metric_name,
      "mean": values.mean(),
      "std": values.std(),
      "min": values.min(),
      "max": values.max(),
      "median": values.median(),
    })

  return pd.DataFrame(rows)


def create_all_data(
  X: pd.DataFrame,
  y: pd.Series,
  target_id: str
) -> pd.DataFrame:
  all_data = X.copy()
  all_data[target_id] = y

  return all_data


def create_class_distribution(
  y: pd.Series
) -> pd.DataFrame:
  counts = (
    y.value_counts()
    .sort_index()
    .rename_axis("class")
    .reset_index(name="count")
  )

  counts["ratio"] = (
    counts["count"] / len(y)
  )

  return counts


def run_cross_validation(
  X: pd.DataFrame,
  y: pd.Series
) -> dict[str, pd.DataFrame]:
  splitter = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE
  )

  metric_rows = []
  prediction_frames = []
  training_rows = []

  for split_number, (
    train_index,
    test_index
  ) in enumerate(
    splitter.split(X, y),
    start=1
  ):
    repeat_number, fold_number = get_repeat_and_fold(
      split_number
    )

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model = create_model(
      split_number=split_number
    )

    model.fit(
      X_train,
      y_train
    )

    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    fnn = model.named_steps["fnn"]

    metric_rows.append({
      "repeat": repeat_number,
      "fold": fold_number,
      "split_number": split_number,
      "train_count": len(train_index),
      "test_count": len(test_index),
      "accuracy": accuracy_score(
        y_test,
        predicted
      ),
      "balanced_accuracy": balanced_accuracy_score(
        y_test,
        predicted
      ),
      "roc_auc": calculate_roc_auc(
        y_true=y_test,
        probabilities=probabilities,
        classes=fnn.classes_
      ),
      "precision_macro": precision_score(
        y_test,
        predicted,
        average="macro",
        zero_division=0
      ),
      "recall_macro": recall_score(
        y_test,
        predicted,
        average="macro",
        zero_division=0
      ),
      "f1_macro": f1_score(
        y_test,
        predicted,
        average="macro",
        zero_division=0
      ),
      "precision_weighted": precision_score(
        y_test,
        predicted,
        average="weighted",
        zero_division=0
      ),
      "recall_weighted": recall_score(
        y_test,
        predicted,
        average="weighted",
        zero_division=0
      ),
      "f1_weighted": f1_score(
        y_test,
        predicted,
        average="weighted",
        zero_division=0
      ),
    })

    training_rows.append({
      "repeat": repeat_number,
      "fold": fold_number,
      "split_number": split_number,
      "iterations": fnn.n_iter_,
      "final_loss": fnn.loss_,
      "best_validation_score": getattr(
        fnn,
        "best_validation_score_",
        np.nan
      ),
    })

    predictions = pd.DataFrame({
      "repeat": repeat_number,
      "fold": fold_number,
      "split_number": split_number,
      "filtered_row": test_index + 1,
      "actual": y_test.to_numpy(),
      "predicted": predicted,
      "correct": (
        y_test.to_numpy() == predicted
      ).astype(int),
    })

    for class_index, class_label in enumerate(
      fnn.classes_
    ):
      predictions[
        f"probability_class_{class_label}"
      ] = probabilities[:, class_index]

    prediction_frames.append(predictions)

  fold_metrics = pd.DataFrame(
    metric_rows
  )

  training_history = pd.DataFrame(
    training_rows
  )

  predictions = pd.concat(
    prediction_frames,
    ignore_index=True
  )

  predictions = predictions.sort_values(
    [
      "repeat",
      "fold",
      "filtered_row"
    ]
  ).reset_index(drop=True)

  summary_metrics = create_metric_summary(
    fold_metrics
  )

  labels = np.sort(
    y.unique()
  )

  report = pd.DataFrame(
    classification_report(
      predictions["actual"],
      predictions["predicted"],
      labels=labels,
      output_dict=True,
      zero_division=0
    )
  ).transpose()

  report = (
    report
    .rename_axis("class")
    .reset_index()
  )

  confusion = pd.DataFrame(
    confusion_matrix(
      predictions["actual"],
      predictions["predicted"],
      labels=labels
    ),
    index=[
      f"actual_{label}"
      for label in labels
    ],
    columns=[
      f"predicted_{label}"
      for label in labels
    ]
  )

  confusion = (
    confusion
    .rename_axis("actual_class")
    .reset_index()
  )

  return {
    "summary_metrics": summary_metrics,
    "fold_metrics": fold_metrics,
    "predictions": predictions,
    "classification_report": report,
    "confusion_matrix": confusion,
    "training_info": training_history,
  }


def save_result(
  output_path: Path,
  config: dict,
  all_data: pd.DataFrame,
  class_distribution: pd.DataFrame,
  result_sheets: dict[str, pd.DataFrame]
) -> None:
  output_path.parent.mkdir(
    parents=True,
    exist_ok=True
  )

  target_config = config["target"]

  run_info = pd.DataFrame({
    "item": [
      "experiment_name",
      "model",
      "task_type",
      "target",
      "ratios",
      "cutoffs",
      "sample_count",
      "feature_count",
      "class_count",
      "n_splits",
      "n_repeats",
      "total_fits",
      "random_state",
      "fnn_params",
      "missing_value_processing",
      "scaling",
    ],
    "value": [
      config["experiment_name"],
      "MLPClassifier",
      "classification",
      target_config["id"],
      json.dumps(
        target_config.get("ratios"),
        ensure_ascii=False
      ),
      json.dumps(
        target_config.get("cutoffs"),
        ensure_ascii=False
      ),
      len(all_data),
      len(config["features"]),
      len(class_distribution),
      N_SPLITS,
      N_REPEATS,
      N_SPLITS * N_REPEATS,
      RANDOM_STATE,
      json.dumps(
        FNN_PARAMS,
        ensure_ascii=False
      ),
      "median imputation within each fold",
      "StandardScaler within each fold",
    ]
  })

  with pd.ExcelWriter(
    output_path,
    engine="openpyxl"
  ) as writer:
    run_info.to_excel(
      writer,
      sheet_name="run_info",
      index=False
    )

    all_data.to_excel(
      writer,
      sheet_name="all_data",
      index=False
    )

    class_distribution.to_excel(
      writer,
      sheet_name="class_distribution",
      index=False
    )

    for sheet_name, data in result_sheets.items():
      data.to_excel(
        writer,
        sheet_name=sheet_name,
        index=False
      )


def run_fnn(
  config_path: str | Path,
  data_path: str | Path | None = None
) -> Path:
  if data_path is None:
    X, y, config = filter_data(
      config_path
    )
  else:
    X, y, config = filter_data(
      config_path,
      data_path
    )

  target_config = config["target"]

  is_classification = (
    target_config.get("ratios") is not None
    or target_config.get("cutoffs") is not None
  )

  if not is_classification:
    raise ValueError(
      "FNN分類を行うには、targetにratiosまたは"
      "cutoffsを指定してください。"
    )

  target_id = str(
    target_config["id"]
  )

  all_data = create_all_data(
    X=X,
    y=y,
    target_id=target_id
  )

  class_distribution = create_class_distribution(
    y
  )

  result_sheets = run_cross_validation(
    X=X,
    y=y
  )

  experiment_name = config[
    "experiment_name"
  ]

  output_path = (
    RESULT_DIR
    / f"result_{experiment_name}_fnn.xlsx"
  )

  save_result(
    output_path=output_path,
    config=config,
    all_data=all_data,
    class_distribution=class_distribution,
    result_sheets=result_sheets
  )

  print(f"実験名: {experiment_name}")
  print("処理: FNN分類")
  print(f"被験者数: {len(X)}")
  print(f"説明変数数: {X.shape[1]}")
  print(f"クラス数: {y.nunique()}")
  print(f"分割数: {N_SPLITS}")
  print(f"反復回数: {N_REPEATS}")
  print(f"学習回数: {N_SPLITS * N_REPEATS}")

  print("\nクラスごとの被験者数:")
  print(
    class_distribution.to_string(
      index=False
    )
  )

  print("\n評価指標:")
  print(
    result_sheets["summary_metrics"][
      ["metric", "mean", "std"]
    ].to_string(index=False)
  )

  print(f"\n保存先: {output_path}")

  return output_path


def main() -> None:
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "config",
    help="設定JSONのパス"
  )

  parser.add_argument(
    "--data",
    default=None,
    help="元データのExcelファイル"
  )

  args = parser.parse_args()

  run_fnn(
    config_path=args.config,
    data_path=args.data
  )


if __name__ == "__main__":
  main()