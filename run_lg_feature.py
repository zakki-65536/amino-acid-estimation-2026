from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from filter import filter_data
from run_lightgbm import (
  evaluate_lightgbm,
  save_result,
)


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"


def get_metric(
  summary_metrics: pd.DataFrame,
  metric_name: str,
  statistic: str
) -> float:
  row = summary_metrics[
    summary_metrics["metric"] == metric_name
  ]

  return float(
    row.iloc[0][statistic]
  )


def run_feature_elimination(
  config_path: str | Path,
  data_path: str | Path | None = None,
  min_features: int = 10,
  output_path: str | Path | None = None
) -> Path:
  if data_path is None:
    X_all, y, config = filter_data(
      config_path
    )
  else:
    X_all, y, config = filter_data(
      config_path,
      data_path
    )

  experiment_name = config[
    "experiment_name"
  ]

  if output_path is None:
    output_dir = (
      RESULT_DIR
      / f"項目削減_{experiment_name}"
    )
  else:
    output_dir = Path(
      output_path
    )

    if not output_dir.is_absolute():
      output_dir = (
        BASE_DIR / output_dir
      )

  output_dir.mkdir(
    parents=True,
    exist_ok=True
  )

  current_features = (
    X_all.columns
    .astype(str)
    .tolist()
  )

  summary_rows = []
  removed_rows = []
  importance_frames = []
  fold_metric_frames = []
  feature_set_frames = []

  iteration = 1

  while len(current_features) >= min_features:
    feature_count = len(
      current_features
    )

    print(
      f"\n===== {feature_count}項目 ====="
    )

    current_config = config.copy()

    current_config["features"] = (
      current_features.copy()
    )

    X_current = X_all[
      current_features
    ].copy()

    evaluation = evaluate_lightgbm(
      X_original=X_current,
      y=y,
      config=current_config
    )

    result_sheets = evaluation[
      "result_sheets"
    ]

    feature_importance = evaluation[
      "feature_importance"
    ].copy()

    iteration_output_path = (
      output_dir
      / f"result_{feature_count}項目.xlsx"
    )

    save_result(
      output_path=iteration_output_path,
      config=current_config,
      all_data=evaluation["all_data"],
      task_type=evaluation["task_type"],
      result_sheets=result_sheets,
      feature_importance=feature_importance
    )

    print(
      f"結果保存: {iteration_output_path}"
    )

    summary_metrics = result_sheets[
      "summary_metrics"
    ]

    auc_mean = get_metric(
      summary_metrics,
      "roc_auc",
      "mean"
    )

    auc_std = get_metric(
      summary_metrics,
      "roc_auc",
      "std"
    )

    accuracy_mean = get_metric(
      summary_metrics,
      "accuracy",
      "mean"
    )

    accuracy_std = get_metric(
      summary_metrics,
      "accuracy",
      "std"
    )

    balanced_accuracy_mean = get_metric(
      summary_metrics,
      "balanced_accuracy",
      "mean"
    )

    balanced_accuracy_std = get_metric(
      summary_metrics,
      "balanced_accuracy",
      "std"
    )

    f1_mean = get_metric(
      summary_metrics,
      "f1",
      "mean"
    )

    f1_std = get_metric(
      summary_metrics,
      "f1",
      "std"
    )

    threshold_summary = result_sheets[
      "threshold_summary"
    ]

    threshold_mean = float(
      threshold_summary.iloc[0]["mean"]
    )

    threshold_std = float(
      threshold_summary.iloc[0]["std"]
    )

    lowest = (
      feature_importance
      .sort_values(
        [
          "mean_importance",
          "feature"
        ],
        ascending=[
          True,
          True
        ]
      )
      .iloc[0]
    )

    lowest_feature = str(
      lowest["feature"]
    )

    lowest_importance = float(
      lowest["mean_importance"]
    )

    feature_to_remove = (
      lowest_feature
      if feature_count > min_features
      else None
    )

    summary_rows.append({
      "iteration": iteration,
      "feature_count": feature_count,
      "roc_auc_mean": auc_mean,
      "roc_auc_std": auc_std,
      "accuracy_mean": accuracy_mean,
      "accuracy_std": accuracy_std,
      "balanced_accuracy_mean": (
        balanced_accuracy_mean
      ),
      "balanced_accuracy_std": (
        balanced_accuracy_std
      ),
      "f1_mean": f1_mean,
      "f1_std": f1_std,
      "threshold_mean": threshold_mean,
      "threshold_std": threshold_std,
      "feature_to_remove": (
        feature_to_remove
      ),
      "removed_mean_importance": (
        lowest_importance
        if feature_to_remove is not None
        else None
      ),
    })

    feature_importance.insert(
      0,
      "feature_count",
      feature_count
    )

    importance_frames.append(
      feature_importance
    )

    fold_metrics = result_sheets[
      "fold_metrics"
    ].copy()

    fold_metrics.insert(
      0,
      "feature_count",
      feature_count
    )

    fold_metric_frames.append(
      fold_metrics
    )

    feature_set_frames.append(
      pd.DataFrame({
        "feature_count": feature_count,
        "feature": current_features,
      })
    )

    print(
      f"AUC: {auc_mean:.6f} "
      f"± {auc_std:.6f}"
    )

    print(
      "Balanced Accuracy: "
      f"{balanced_accuracy_mean:.6f}"
    )

    if feature_to_remove is None:
      break

    print(
      f"削除: {lowest_feature} "
      f"(importance={lowest_importance:.6f})"
    )

    removed_rows.append({
      "iteration": iteration,
      "feature_count_before": (
        feature_count
      ),
      "removed_feature": (
        lowest_feature
      ),
      "mean_importance": (
        lowest_importance
      ),
    })

    current_features.remove(
      lowest_feature
    )

    iteration += 1

  summary = pd.DataFrame(
    summary_rows
  )

  removed_features = pd.DataFrame(
    removed_rows
  )

  importance_history = pd.concat(
    importance_frames,
    ignore_index=True
  )

  fold_metrics_history = pd.concat(
    fold_metric_frames,
    ignore_index=True
  )

  feature_sets = pd.concat(
    feature_set_frames,
    ignore_index=True
  )

  best_index = (
    summary["roc_auc_mean"].idxmax()
  )

  best_result = summary.loc[
    [best_index]
  ].copy()

  best_feature_count = int(
    best_result.iloc[0][
      "feature_count"
    ]
  )

  best_features = (
    feature_sets[
      feature_sets["feature_count"]
      == best_feature_count
    ]
    .copy()
  )

  summary_output_path = (
    output_dir
    / "summary.xlsx"
  )

  with pd.ExcelWriter(
    summary_output_path,
    engine="openpyxl"
  ) as writer:
    summary.to_excel(
      writer,
      sheet_name="summary",
      index=False
    )

    best_result.to_excel(
      writer,
      sheet_name="best_result",
      index=False
    )

    best_features.to_excel(
      writer,
      sheet_name="best_features",
      index=False
    )

    removed_features.to_excel(
      writer,
      sheet_name="removed_features",
      index=False
    )

    feature_sets.to_excel(
      writer,
      sheet_name="feature_sets",
      index=False
    )

    importance_history.to_excel(
      writer,
      sheet_name="importance_history",
      index=False
    )

    fold_metrics_history.to_excel(
      writer,
      sheet_name="fold_metrics",
      index=False
    )

  print(
    "\n===== 最良結果 ====="
  )

  print(
    f"項目数: {best_feature_count}"
  )

  print(
    "AUC: "
    f"{best_result.iloc[0]['roc_auc_mean']:.6f} "
    "± "
    f"{best_result.iloc[0]['roc_auc_std']:.6f}"
  )

  print(
    f"\n集計結果保存先: {summary_output_path}"
  )

  return summary_output_path


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

  parser.add_argument(
    "--min-features",
    type=int,
    default=10,
    help="何項目まで削減するか"
  )

  parser.add_argument(
    "--output",
    default=None,
    help="結果Excelの保存先"
  )

  args = parser.parse_args()

  run_feature_elimination(
    config_path=args.config,
    data_path=args.data,
    min_features=args.min_features,
    output_path=args.output
  )


if __name__ == "__main__":
  main()