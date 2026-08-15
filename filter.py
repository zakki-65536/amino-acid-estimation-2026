from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "data.xlsx"


def normalize_item_id(value: object) -> str:
  if isinstance(value, float) and value.is_integer():
    return str(int(value))

  return str(value).strip()


def resolve_path(path: str | Path) -> Path:
  path = Path(path)

  if path.is_absolute():
    return path

  return BASE_DIR / path


def load_config(config_path: str | Path) -> dict:
  path = resolve_path(config_path)

  with path.open("r", encoding="utf-8") as file:
    return json.load(file)


def load_data(
  data_path: str | Path = DEFAULT_DATA_PATH
) -> pd.DataFrame:
  raw = pd.read_excel(
    resolve_path(data_path),
    header=None
  )

  item_ids = [
    normalize_item_id(value)
    for value in raw.iloc[0]
  ]

  data = raw.iloc[2:].copy()
  data.columns = item_ids
  data = data.dropna(how="all").reset_index(drop=True)

  return data.apply(
    pd.to_numeric,
    errors="coerce"
  )


def calculate_class_counts(
  sample_count: int,
  ratios: list[float]
) -> list[int]:
  ratio_array = np.asarray(
    ratios,
    dtype=float
  )

  exact_counts = (
    sample_count
    * ratio_array
    / ratio_array.sum()
  )

  counts = np.floor(
    exact_counts
  ).astype(int)

  remainder = sample_count - counts.sum()

  remainder_order = np.argsort(
    -(exact_counts - counts)
  )

  counts[
    remainder_order[:remainder]
  ] += 1

  return counts.tolist()


def discretize_by_ratios(
  target: pd.Series,
  ratios: list[float]
) -> pd.Series:
  class_counts = calculate_class_counts(
    sample_count=len(target),
    ratios=ratios
  )

  sorted_indexes = target.sort_values(
    kind="mergesort"
  ).index

  class_labels = np.concatenate([
    np.full(count, class_number)
    for class_number, count in enumerate(class_counts)
  ])

  discretized = pd.Series(
    index=target.index,
    dtype=int
  )

  discretized.loc[sorted_indexes] = class_labels

  return discretized.astype(int)


def discretize_by_cutoffs(
  target: pd.Series,
  cutoffs: list[float]
) -> pd.Series:
  return pd.Series(
    np.digitize(
      target.to_numpy(),
      bins=cutoffs,
      right=False
    ),
    index=target.index,
    dtype=int
  )


def apply_filters(
  data: pd.DataFrame,
  filters: list[dict]
) -> pd.DataFrame:
  filtered = data

  for condition in filters:
    item_id = normalize_item_id(
      condition["id"]
    )

    filtered = filtered[
      filtered[item_id] == condition["value"]
    ]

  return filtered.reset_index(drop=True)


def filter_data(
  config_path: str | Path,
  data_path: str | Path = DEFAULT_DATA_PATH
) -> tuple[pd.DataFrame, pd.Series, dict]:
  config = load_config(config_path)
  data = load_data(data_path)

  data = apply_filters(
    data=data,
    filters=config["filters"]
  )

  target_config = config["target"]

  target_id = normalize_item_id(
    target_config["id"]
  )

  feature_ids = [
    normalize_item_id(item_id)
    for item_id in config["features"]
  ]

  X = data[feature_ids].copy()
  y = data[target_id].copy()

  ratios = target_config.get("ratios")
  cutoffs = target_config.get("cutoffs")

  if ratios is not None and cutoffs is not None:
    raise ValueError(
      "ratiosとcutoffsは同時に指定できません。"
    )

  if ratios is not None:
    y = discretize_by_ratios(
      target=y,
      ratios=ratios
    )

  elif cutoffs is not None:
    y = discretize_by_cutoffs(
      target=y,
      cutoffs=cutoffs
    )

  X = X.reset_index(drop=True)
  y = y.reset_index(drop=True)

  return X, y, config


def main() -> None:
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "config",
    help="設定JSONのパス"
  )

  parser.add_argument(
    "--data",
    default=str(DEFAULT_DATA_PATH),
    help="元データのExcelファイル"
  )

  args = parser.parse_args()

  X, y, config = filter_data(
    config_path=args.config,
    data_path=args.data
  )

  print(f"実験名: {config['experiment_name']}")
  print(f"被験者数: {len(X)}")
  print(f"説明変数数: {X.shape[1]}")

  target_config = config["target"]

  if target_config.get("ratios") is not None:
    print(f"分類方法: ratios {target_config['ratios']}")
    print(y.value_counts().sort_index())

  elif target_config.get("cutoffs") is not None:
    print(f"分類方法: cutoffs {target_config['cutoffs']}")
    print(y.value_counts().sort_index())

  else:
    print("目的変数: 連続値")


if __name__ == "__main__":
  main()