from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
  accuracy_score,
  balanced_accuracy_score,
  classification_report,
  confusion_matrix,
  f1_score,
  mean_absolute_error,
  mean_squared_error,
  precision_score,
  r2_score,
  recall_score,
  roc_auc_score,
)
from sklearn.model_selection import (
  RepeatedKFold,
  RepeatedStratifiedKFold,
  StratifiedKFold,
  TunedThresholdClassifierCV,
)
from filter import filter_data


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"

N_SPLITS = 5
N_REPEATS = 10
INNER_N_SPLITS = 5
RANDOM_STATE = 42

THRESHOLD_SCORING = "balanced_accuracy"
THRESHOLD_COUNT = 100

MODEL_PARAMS = {
  "n_estimators": 300,
  "learning_rate": 0.03,
  "num_leaves": 15,
  "min_child_samples": 10,
  "subsample": 0.8,
  "subsample_freq": 1,
  "colsample_bytree": 0.8,
  "random_state": RANDOM_STATE,
  "n_jobs": -1,
  "verbosity": -1,
}


def rename_features_for_lightgbm(
  X: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, str]]:
  original_names = X.columns.astype(str).tolist()

  safe_names = [
    f"feature_{index}"
    for index in range(len(original_names))
  ]

  feature_name_map = dict(
    zip(safe_names, original_names)
  )

  renamed_X = X.copy()
  renamed_X.columns = safe_names

  return renamed_X, feature_name_map


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


def create_metric_summary(
  fold_metrics: pd.DataFrame,
  metric_names: list[str]
) -> pd.DataFrame:
  summary_rows = []

  for metric_name in metric_names:
    values = fold_metrics[metric_name]

    summary_rows.append({
      "metric": metric_name,
      "mean": values.mean(),
      "std": values.std(),
      "min": values.min(),
      "max": values.max(),
      "median": values.median(),
    })

  return pd.DataFrame(summary_rows)


def create_feature_importance_summary(
  importance_data: pd.DataFrame
) -> pd.DataFrame:
  feature_importance = (
    importance_data
    .groupby(
      "feature",
      as_index=False
    )
    .agg(
      mean_importance=("importance", "mean"),
      std_importance=("importance", "std"),
      min_importance=("importance", "min"),
      max_importance=("importance", "max"),
    )
    .sort_values(
      "mean_importance",
      ascending=False
    )
    .reset_index(drop=True)
  )

  feature_importance.insert(
    0,
    "rank",
    range(1, len(feature_importance) + 1)
  )

  return feature_importance


def run_classification(
  X: pd.DataFrame,
  y: pd.Series,
  feature_name_map: dict[str, str]
) -> tuple[
  dict[str, pd.DataFrame],
  pd.DataFrame
]:
  outer_cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE
  )

  is_binary = y.nunique() == 2

  metric_rows = []
  prediction_frames = []
  importance_frames = []
  threshold_curve_frames = []

  for split_number, (
    train_index,
    test_index
  ) in enumerate(
    outer_cv.split(X, y),
    start=1
  ):
    repeat_number, fold_number = get_repeat_and_fold(
      split_number
    )

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    if is_binary:
      inner_cv = StratifiedKFold(
        n_splits=INNER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE + split_number
      )

      base_model = LGBMClassifier(
        **MODEL_PARAMS
      )

      model = TunedThresholdClassifierCV(
        estimator=base_model,
        scoring=THRESHOLD_SCORING,
        response_method="predict_proba",
        thresholds=THRESHOLD_COUNT,
        cv=inner_cv,
        refit=True,
        n_jobs=1,
        store_cv_results=True
      )

      model.fit(
        X_train,
        y_train
      )

      predicted = model.predict(X_test)
      probabilities = model.predict_proba(X_test)

      fitted_lightgbm = model.estimator_

      selected_threshold = model.best_threshold_
      inner_best_score = model.best_score_

    else:
      model = LGBMClassifier(
        **MODEL_PARAMS
      )

      model.fit(
        X_train,
        y_train
      )

      predicted = model.predict(X_test)
      probabilities = model.predict_proba(X_test)

      fitted_lightgbm = model

      selected_threshold = np.nan
      inner_best_score = np.nan

    metric_row = {
      "repeat": repeat_number,
      "fold": fold_number,
      "split_number": split_number,
      "train_count": len(train_index),
      "test_count": len(test_index),
      "selected_threshold": selected_threshold,
      "inner_best_score": inner_best_score,
      "accuracy": accuracy_score(
        y_test,
        predicted
      ),
      "balanced_accuracy": balanced_accuracy_score(
        y_test,
        predicted
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
    }

    if is_binary:
      metric_row["roc_auc"] = roc_auc_score(
        y_test,
        probabilities[:, 1]
      )

      negative_label = model.classes_[0]
      positive_label = model.classes_[1]

      metric_row["precision"] = precision_score(
        y_test,
        predicted,
        pos_label=positive_label,
        zero_division=0
      )

      metric_row["sensitivity"] = recall_score(
        y_test,
        predicted,
        pos_label=positive_label,
        zero_division=0
      )

      metric_row["specificity"] = recall_score(
        y_test,
        predicted,
        pos_label=negative_label,
        zero_division=0
      )

      metric_row["f1"] = f1_score(
        y_test,
        predicted,
        pos_label=positive_label,
        zero_division=0
      )

    else:
      metric_row["roc_auc_ovr_macro"] = roc_auc_score(
        y_test,
        probabilities,
        labels=model.classes_,
        multi_class="ovr",
        average="macro"
      )

    metric_rows.append(metric_row)

    predictions = pd.DataFrame({
      "repeat": repeat_number,
      "fold": fold_number,
      "split_number": split_number,
      "filtered_row": test_index + 1,
      "actual": y_test.to_numpy(),
      "predicted": predicted,
    })

    if is_binary:
      predictions["selected_threshold"] = (
        selected_threshold
      )

    for class_index, class_label in enumerate(
      model.classes_
    ):
      predictions[
        f"probability_class_{class_label}"
      ] = probabilities[:, class_index]

    prediction_frames.append(predictions)

    importance_frames.append(
      pd.DataFrame({
        "repeat": repeat_number,
        "fold": fold_number,
        "split_number": split_number,
        "feature": [
          feature_name_map[name]
          for name in fitted_lightgbm.feature_name_
        ],
        "importance": (
          fitted_lightgbm.feature_importances_
        ),
      })
    )

    if is_binary:
      threshold_curve_frames.append(
        pd.DataFrame({
          "repeat": repeat_number,
          "fold": fold_number,
          "split_number": split_number,
          "threshold": (
            model.cv_results_["thresholds"]
          ),
          "inner_cv_score": (
            model.cv_results_["scores"]
          ),
        })
      )

  fold_metrics = pd.DataFrame(
    metric_rows
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

  importance_data = pd.concat(
    importance_frames,
    ignore_index=True
  )

  feature_importance = (
    create_feature_importance_summary(
      importance_data
    )
  )

  if is_binary:
    metric_names = [
      "accuracy",
      "balanced_accuracy",
      "precision",
      "sensitivity",
      "specificity",
      "f1",
      "roc_auc",
      "precision_macro",
      "recall_macro",
      "f1_macro",
    ]
  else:
    metric_names = [
      "accuracy",
      "balanced_accuracy",
      "precision_macro",
      "recall_macro",
      "f1_macro",
      "roc_auc_ovr_macro",
    ]

  summary_metrics = create_metric_summary(
    fold_metrics=fold_metrics,
    metric_names=metric_names
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

  result_sheets = {
    "summary_metrics": summary_metrics,
    "fold_metrics": fold_metrics,
    "predictions": predictions,
    "classification_report": report,
    "confusion_matrix": confusion,
  }

  if is_binary:
    thresholds = fold_metrics[
      "selected_threshold"
    ]

    result_sheets["threshold_summary"] = (
      pd.DataFrame([{
        "scoring": THRESHOLD_SCORING,
        "mean": thresholds.mean(),
        "std": thresholds.std(),
        "min": thresholds.min(),
        "max": thresholds.max(),
        "median": thresholds.median(),
      }])
    )

    result_sheets["threshold_curves"] = (
      pd.concat(
        threshold_curve_frames,
        ignore_index=True
      )
    )

  return result_sheets, feature_importance


def run_regression(
  X: pd.DataFrame,
  y: pd.Series,
  feature_name_map: dict[str, str]
) -> tuple[
  dict[str, pd.DataFrame],
  pd.DataFrame
]:
  splitter = RepeatedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE
  )

  metric_rows = []
  prediction_frames = []
  importance_frames = []

  for split_number, (
    train_index,
    test_index
  ) in enumerate(
    splitter.split(X),
    start=1
  ):
    repeat_number, fold_number = get_repeat_and_fold(
      split_number
    )

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    model = LGBMRegressor(
      **MODEL_PARAMS
    )

    model.fit(
      X_train,
      y_train
    )

    predicted = model.predict(X_test)

    metric_rows.append({
      "repeat": repeat_number,
      "fold": fold_number,
      "split_number": split_number,
      "train_count": len(train_index),
      "test_count": len(test_index),
      "mae": mean_absolute_error(
        y_test,
        predicted
      ),
      "rmse": np.sqrt(
        mean_squared_error(
          y_test,
          predicted
        )
      ),
      "r2": r2_score(
        y_test,
        predicted
      ),
    })

    prediction_frames.append(
      pd.DataFrame({
        "repeat": repeat_number,
        "fold": fold_number,
        "split_number": split_number,
        "filtered_row": test_index + 1,
        "actual": y_test.to_numpy(),
        "predicted": predicted,
        "absolute_error": np.abs(
          y_test.to_numpy() - predicted
        ),
        "squared_error": np.square(
          y_test.to_numpy() - predicted
        ),
      })
    )

    importance_frames.append(
      pd.DataFrame({
        "repeat": repeat_number,
        "fold": fold_number,
        "split_number": split_number,
        "feature": [
          feature_name_map[name]
          for name in model.feature_name_
        ],
        "importance": model.feature_importances_,
      })
    )

  fold_metrics = pd.DataFrame(
    metric_rows
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

  importance_data = pd.concat(
    importance_frames,
    ignore_index=True
  )

  feature_importance = (
    create_feature_importance_summary(
      importance_data
    )
  )

  summary_metrics = create_metric_summary(
    fold_metrics=fold_metrics,
    metric_names=[
      "mae",
      "rmse",
      "r2",
    ]
  )

  result_sheets = {
    "summary_metrics": summary_metrics,
    "fold_metrics": fold_metrics,
    "predictions": predictions,
  }

  return result_sheets, feature_importance


def create_all_data(
  X: pd.DataFrame,
  y: pd.Series,
  target_id: str
) -> pd.DataFrame:
  all_data = X.copy()
  all_data[target_id] = y

  return all_data


def save_result(
  output_path: Path,
  config: dict,
  all_data: pd.DataFrame,
  task_type: str,
  result_sheets: dict[str, pd.DataFrame],
  feature_importance: pd.DataFrame
) -> None:
  output_path.parent.mkdir(
    parents=True,
    exist_ok=True
  )

  target_config = config["target"]

  run_info = pd.DataFrame({
    "item": [
      "experiment_name",
      "task_type",
      "target",
      "ratios",
      "cutoffs",
      "sample_count",
      "feature_count",
      "n_splits",
      "n_repeats",
      "total_fits",
      "random_state",
      "model_params",
    ],
    "value": [
      config["experiment_name"],
      task_type,
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
      N_SPLITS,
      N_REPEATS,
      N_SPLITS * N_REPEATS,
      RANDOM_STATE,
      json.dumps(
        MODEL_PARAMS,
        ensure_ascii=False
      ),
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
      sheet_name="data",
      index=False
    )

    for sheet_name, data in result_sheets.items():
      data.to_excel(
        writer,
        sheet_name=sheet_name,
        index=False
      )

    feature_importance.to_excel(
      writer,
      sheet_name="feature_importance",
      index=False
    )


def run_lightgbm(
  config_path: str | Path,
  data_path: str | Path | None = None
) -> Path:
  if data_path is None:
    X_original, y, config = filter_data(
      config_path
    )
  else:
    X_original, y, config = filter_data(
      config_path,
      data_path
    )

  target_config = config["target"]

  all_data = create_all_data(
    X=X_original,
    y=y,
    target_id=str(target_config["id"])
  )

  X, feature_name_map = rename_features_for_lightgbm(
    X_original
  )

  is_classification = (
    target_config.get("ratios") is not None
    or target_config.get("cutoffs") is not None
  )

  if is_classification:
    result_sheets, feature_importance = (
      run_classification(
        X=X,
        y=y,
        feature_name_map=feature_name_map
      )
    )

    task_type = "classification"

  else:
    result_sheets, feature_importance = (
      run_regression(
        X=X,
        y=y,
        feature_name_map=feature_name_map
      )
    )

    task_type = "regression"

  experiment_name = config[
    "experiment_name"
  ]

  output_path = (
    RESULT_DIR
    / f"result_{experiment_name}.xlsx"
  )

  save_result(
    output_path=output_path,
    config=config,
    all_data=all_data,
    task_type=task_type,
    result_sheets=result_sheets,
    feature_importance=feature_importance
  )

  print(f"実験名: {experiment_name}")
  print(f"処理: {task_type}")
  print(f"被験者数: {len(X)}")
  print(f"説明変数数: {X.shape[1]}")
  print(f"分割数: {N_SPLITS}")
  print(f"反復回数: {N_REPEATS}")
  print(f"学習回数: {N_SPLITS * N_REPEATS}")

  print("\n評価指標:")
  print(
    result_sheets["summary_metrics"][
      ["metric", "mean", "std"]
    ].to_string(index=False)
  )

  print(f"\n保存先: {output_path}")
  
  if "threshold_summary" in result_sheets:
    threshold_summary = result_sheets[
      "threshold_summary"
    ]

    print(
      "\n選択された閾値:"
    )

    print(
      threshold_summary.to_string(
        index=False
      )
    )

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

  run_lightgbm(
    config_path=args.config,
    data_path=args.data
  )


if __name__ == "__main__":
  main()