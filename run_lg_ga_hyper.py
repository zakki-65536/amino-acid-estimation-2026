from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from filter import filter_data
from run_lightgbm import (
    DEFAULT_THRESHOLD,
    evaluate_lightgbm,
    evaluate_lightgbm_holdout,
)


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"

# =========================
# Default GA / CV settings
# =========================
OUTER_N_SPLITS = 5
OUTER_N_REPEATS = 10
OUTER_RANDOM_STATE = 42

INNER_N_SPLITS = 5
INNER_N_REPEATS = 1

POPULATION_SIZE = 40
GENERATIONS = 200
FITNESS_LAMBDA = 0.5

CROSSOVER_RATE = 0.8
ELITE_SIZE = 2
TOURNAMENT_SIZE = 3

GA_THREAD_COUNT = 32
RANDOM_SEED = 42

# ============================================================
# Hyperparameter candidate space
# 12 * 16 * 6 * 4 * 4 * 4 * 4 = 294,912 combinations
# ============================================================

ESTIMATOR_LR_PAIRS: tuple[tuple[int, float], ...] = (
    (100, 0.15),
    (150, 0.15),
    (200, 0.10),
    (250, 0.10),
    (400, 0.07),
    (400, 0.10),
    (600, 0.05),
    (600, 0.03),
    (800, 0.03),
    (800, 0.05),
    (1000, 0.02),
    (1200, 0.02),
)

DEPTH_LEAVES_PAIRS: tuple[tuple[int, int], ...] = (
    (3, 3),
    (3, 7),
    (4, 7),
    (4, 15),
    (6, 7),
    (6, 15),
    (6, 31),
    (6, 63),
    (8, 7),
    (8, 15),
    (8, 31),
    (8, 63),
    (8, 127),
    (10, 31),
    (10, 63),
    (10, 127),
)

MIN_CHILD_SAMPLES_VALUES: tuple[int, ...] = (
    5,
    10,
    20,
    30,
    60,
    100,
)

SUBSAMPLE_VALUES: tuple[float, ...] = (
    0.7,
    0.8,
    0.9,
    1.0,
)

COLSAMPLE_BYTREE_VALUES: tuple[float, ...] = (
    0.7,
    0.8,
    0.9,
    1.0,
)

REG_ALPHA_VALUES: tuple[float, ...] = (
    0.0,
    0.1,
    1.0,
    10.0,
)

REG_LAMBDA_VALUES: tuple[float, ...] = (
    0.0,
    0.1,
    1.0,
    10.0,
)

# 7 parameter genes:
# 0: n_estimators x learning_rate pair
# 1: max_depth x num_leaves pair
# 2: min_child_samples
# 3: subsample
# 4: colsample_bytree
# 5: reg_alpha
# 6: reg_lambda
PARAMETER_GENE_LENGTHS: tuple[int, ...] = (
    len(ESTIMATOR_LR_PAIRS),
    len(DEPTH_LEAVES_PAIRS),
    len(MIN_CHILD_SAMPLES_VALUES),
    len(SUBSAMPLE_VALUES),
    len(COLSAMPLE_BYTREE_VALUES),
    len(REG_ALPHA_VALUES),
    len(REG_LAMBDA_VALUES),
)

PARAMETER_GENE_COUNT = len(PARAMETER_GENE_LENGTHS)

PARAMETER_COMBINATION_COUNT = int(
    np.prod(PARAMETER_GENE_LENGTHS)
)

if PARAMETER_COMBINATION_COUNT != 294_912:
    raise RuntimeError(
        "ハイパーパラメータ候補数が294,912通りになっていません。"
    )


@dataclass
class Individual:
    feature_mask: np.ndarray
    parameter_genes: np.ndarray

    def copy(self) -> "Individual":
        return Individual(
            feature_mask=self.feature_mask.copy(),
            parameter_genes=self.parameter_genes.copy(),
        )

    def key(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (
            tuple(
                int(x)
                for x in self.feature_mask
            ),
            tuple(
                int(x)
                for x in self.parameter_genes
            ),
        )


@dataclass(frozen=True)
class EvaluationResult:
    fitness: float
    auc_mean: float
    auc_std: float


def current_timestamp() -> str:
    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def get_repeat_and_fold(
    split_number: int,
    n_splits: int,
) -> tuple[int, int]:
    repeat_number = (
        (split_number - 1) // n_splits
    ) + 1

    fold_number = (
        (split_number - 1) % n_splits
    ) + 1

    return repeat_number, fold_number


def selected_features_from_mask(
    feature_mask: np.ndarray,
    all_features: list[str],
) -> list[str]:
    return [
        feature
        for gene, feature in zip(
            feature_mask,
            all_features,
        )
        if gene == 1
    ]


def parameter_genes_to_dict(
    genes: np.ndarray,
) -> dict[str, int | float]:
    estimator_lr_index = int(genes[0])
    depth_leaves_index = int(genes[1])

    (
        n_estimators,
        learning_rate,
    ) = ESTIMATOR_LR_PAIRS[
        estimator_lr_index
    ]

    (
        max_depth,
        num_leaves,
    ) = DEPTH_LEAVES_PAIRS[
        depth_leaves_index
    ]

    return {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "min_child_samples": (
            MIN_CHILD_SAMPLES_VALUES[
                int(genes[2])
            ]
        ),
        "subsample": SUBSAMPLE_VALUES[
            int(genes[3])
        ],
        "colsample_bytree": (
            COLSAMPLE_BYTREE_VALUES[
                int(genes[4])
            ]
        ),
        "reg_alpha": REG_ALPHA_VALUES[
            int(genes[5])
        ],
        "reg_lambda": REG_LAMBDA_VALUES[
            int(genes[6])
        ],
    }


def parameter_genes_to_record(
    genes: np.ndarray,
) -> dict[str, int | float]:
    params = parameter_genes_to_dict(
        genes
    )

    return {
        "estimator_lr_pair_no": (
            int(genes[0]) + 1
        ),
        "depth_leaves_pair_no": (
            int(genes[1]) + 1
        ),
        **params,
    }


def repair_feature_mask(
    feature_mask: np.ndarray,
    min_features: int,
    max_features: int,
    rng: np.random.Generator,
) -> np.ndarray:
    result = feature_mask.copy()

    selected = np.flatnonzero(
        result == 1
    )
    unselected = np.flatnonzero(
        result == 0
    )

    if len(selected) < min_features:
        add_count = (
            min_features
            - len(selected)
        )

        add_indexes = rng.choice(
            unselected,
            size=add_count,
            replace=False,
        )

        result[add_indexes] = 1

    selected = np.flatnonzero(
        result == 1
    )

    if len(selected) > max_features:
        remove_count = (
            len(selected)
            - max_features
        )

        remove_indexes = rng.choice(
            selected,
            size=remove_count,
            replace=False,
        )

        result[remove_indexes] = 0

    return result


def random_parameter_genes(
    rng: np.random.Generator,
) -> np.ndarray:
    return np.asarray(
        [
            int(
                rng.integers(
                    0,
                    length,
                )
            )
            for length
            in PARAMETER_GENE_LENGTHS
        ],
        dtype=np.int16,
    )


def initialize_population(
    population_size: int,
    feature_count: int,
    min_features: int,
    max_features: int,
    rng: np.random.Generator,
) -> list[Individual]:
    population: list[Individual] = []

    # 1個体目は全特徴量 + 各パラメータ候補の中央付近。
    # それ以外は特徴量・パラメータともランダム。
    first_feature_mask = (
        np.ones(
            feature_count,
            dtype=np.int8,
        )
    )

    first_feature_mask = (
        repair_feature_mask(
            first_feature_mask,
            min_features=min_features,
            max_features=max_features,
            rng=rng,
        )
    )

    first_parameter_genes = np.asarray(
        [
            length // 2
            for length
            in PARAMETER_GENE_LENGTHS
        ],
        dtype=np.int16,
    )

    population.append(
        Individual(
            feature_mask=(
                first_feature_mask
            ),
            parameter_genes=(
                first_parameter_genes
            ),
        )
    )

    while (
        len(population)
        < population_size
    ):
        selected_count = int(
            rng.integers(
                min_features,
                max_features + 1,
            )
        )

        feature_mask = np.zeros(
            feature_count,
            dtype=np.int8,
        )

        selected_indexes = rng.choice(
            feature_count,
            size=selected_count,
            replace=False,
        )

        feature_mask[
            selected_indexes
        ] = 1

        population.append(
            Individual(
                feature_mask=(
                    feature_mask
                ),
                parameter_genes=(
                    random_parameter_genes(
                        rng
                    )
                ),
            )
        )

    return population


def get_auc_mean_std(
    evaluation: dict,
) -> tuple[float, float]:
    summary_metrics = evaluation[
        "result_sheets"
    ]["summary_metrics"]

    auc_row = summary_metrics[
        summary_metrics["metric"]
        == "roc_auc"
    ]

    if auc_row.empty:
        raise ValueError(
            "summary_metrics に roc_auc がありません。"
        )

    auc_mean = float(
        auc_row.iloc[0]["mean"]
    )
    auc_std = float(
        auc_row.iloc[0]["std"]
    )

    return auc_mean, auc_std


def build_individual_model_params(
    individual: Individual,
    model_random_state: int,
) -> dict[str, int | float]:
    """
    個体のパラメータ遺伝子を LightGBM の設定へ変換する。

    GAはIndividualを並列評価するため、LightGBM側はn_jobs=1に固定する。
    run_lightgbm.py のグローバル MODEL_PARAMS は変更しない。
    """
    return {
        **parameter_genes_to_dict(
            individual.parameter_genes
        ),
        "objective": "binary",
        "subsample_freq": 1,
        "random_state": model_random_state,
        "n_jobs": 1,
        "verbosity": -1,
    }


def evaluate_individual(
    individual: Individual,
    all_features: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    inner_n_splits: int,
    inner_n_repeats: int,
    fitness_lambda: float,
    model_random_state: int,
) -> EvaluationResult:
    selected_features = (
        selected_features_from_mask(
            individual.feature_mask,
            all_features,
        )
    )

    if not selected_features:
        return EvaluationResult(
            fitness=float("-inf"),
            auc_mean=float("nan"),
            auc_std=float("nan"),
        )

    current_config = copy.deepcopy(
        config
    )
    current_config["features"] = (
        selected_features.copy()
    )

    X_current = X_train[
        selected_features
    ].copy()

    model_params = (
        build_individual_model_params(
            individual=individual,
            model_random_state=(
                model_random_state
            ),
        )
    )

    try:
        evaluation = evaluate_lightgbm(
            X_original=X_current,
            y=y_train,
            config=current_config,
            # FitnessはAUCのみなので閾値探索は常に省略する。
            skip_threshold_cv=True,
            default_threshold=(
                DEFAULT_THRESHOLD
            ),
            n_splits=inner_n_splits,
            n_repeats=inner_n_repeats,
            model_params=model_params,
        )

        auc_mean, auc_std = (
            get_auc_mean_std(
                evaluation
            )
        )

        if (
            not np.isfinite(auc_mean)
            or not np.isfinite(auc_std)
        ):
            raise ValueError(
                "AUCの平均または標準偏差が有限値ではありません。"
            )

        fitness = float(
            auc_mean
            - fitness_lambda
            * auc_std
        )

    except Exception as exc:
        print(
            "    個体評価失敗: "
            f"features="
            f"{len(selected_features)} / "
            f"{exc}"
        )

        return EvaluationResult(
            fitness=float("-inf"),
            auc_mean=float("nan"),
            auc_std=float("nan"),
        )

    return EvaluationResult(
        fitness=fitness,
        auc_mean=auc_mean,
        auc_std=auc_std,
    )


def tournament_selection(
    population: list[Individual],
    fitnesses: list[float],
    tournament_size: int,
    rng: np.random.Generator,
) -> Individual:
    size = min(
        tournament_size,
        len(population),
    )

    indexes = rng.choice(
        len(population),
        size=size,
        replace=False,
    )

    best_index = max(
        indexes,
        key=lambda index: (
            fitnesses[
                int(index)
            ]
        ),
    )

    return population[
        int(best_index)
    ].copy()


def uniform_crossover(
    parent1: Individual,
    parent2: Individual,
    crossover_rate: float,
    rng: np.random.Generator,
) -> tuple[
    Individual,
    Individual,
]:
    if (
        rng.random()
        >= crossover_rate
    ):
        return (
            parent1.copy(),
            parent2.copy(),
        )

    feature_mask = (
        rng.random(
            len(
                parent1.feature_mask
            )
        )
        < 0.5
    )

    child1_features = np.where(
        feature_mask,
        parent1.feature_mask,
        parent2.feature_mask,
    ).astype(
        np.int8
    )

    child2_features = np.where(
        feature_mask,
        parent2.feature_mask,
        parent1.feature_mask,
    ).astype(
        np.int8
    )

    parameter_mask = (
        rng.random(
            PARAMETER_GENE_COUNT
        )
        < 0.5
    )

    child1_parameters = np.where(
        parameter_mask,
        parent1.parameter_genes,
        parent2.parameter_genes,
    ).astype(
        np.int16
    )

    child2_parameters = np.where(
        parameter_mask,
        parent2.parameter_genes,
        parent1.parameter_genes,
    ).astype(
        np.int16
    )

    return (
        Individual(
            feature_mask=(
                child1_features
            ),
            parameter_genes=(
                child1_parameters
            ),
        ),
        Individual(
            feature_mask=(
                child2_features
            ),
            parameter_genes=(
                child2_parameters
            ),
        ),
    )


def mutate_adjacent_parameter_gene(
    current_index: int,
    candidate_count: int,
    rng: np.random.Generator,
) -> int:
    """
    パラメータ突然変異は候補リスト上の隣接値へ移動する。

    先頭 -> 1つ右
    末尾 -> 1つ左
    中間 -> 左右を50%ずつ
    """
    if candidate_count <= 1:
        return current_index

    if current_index <= 0:
        return 1

    if (
        current_index
        >= candidate_count - 1
    ):
        return (
            candidate_count - 2
        )

    direction = (
        -1
        if rng.random() < 0.5
        else 1
    )

    return (
        current_index
        + direction
    )


def mutate_individual(
    individual: Individual,
    feature_mutation_rate: float,
    parameter_mutation_rate: float,
    min_features: int,
    max_features: int,
    rng: np.random.Generator,
) -> Individual:
    result = individual.copy()

    # 各特徴量遺伝子を
    # default 1 / feature_count の確率で反転。
    # したがって期待変異数は1個。
    feature_mutation_mask = (
        rng.random(
            len(
                result.feature_mask
            )
        )
        < feature_mutation_rate
    )

    result.feature_mask[
        feature_mutation_mask
    ] = (
        1
        - result.feature_mask[
            feature_mutation_mask
        ]
    )

    result.feature_mask = (
        repair_feature_mask(
            result.feature_mask,
            min_features=(
                min_features
            ),
            max_features=(
                max_features
            ),
            rng=rng,
        )
    )

    # 7個のパラメータ遺伝子を
    # default 1 / 7 の確率で変異。
    # したがって期待変異数は1個。
    for gene_index, candidate_count in enumerate(
        PARAMETER_GENE_LENGTHS
    ):
        if (
            rng.random()
            < parameter_mutation_rate
        ):
            result.parameter_genes[
                gene_index
            ] = (
                mutate_adjacent_parameter_gene(
                    current_index=int(
                        result.parameter_genes[
                            gene_index
                        ]
                    ),
                    candidate_count=(
                        candidate_count
                    ),
                    rng=rng,
                )
            )

    return result


def individual_to_history_row(
    individual: Individual,
    all_features: list[str],
    evaluation: EvaluationResult,
    generation: int,
    individual_number: int,
    fitness_lambda: float,
) -> dict:
    selected_features = (
        selected_features_from_mask(
            individual.feature_mask,
            all_features,
        )
    )

    parameter_record = (
        parameter_genes_to_record(
            individual.parameter_genes
        )
    )

    return {
        "generation": generation,
        "individual": (
            individual_number
        ),
        "fitness": (
            evaluation.fitness
        ),
        "auc_mean": (
            evaluation.auc_mean
        ),
        "auc_std": (
            evaluation.auc_std
        ),
        "fitness_lambda": (
            fitness_lambda
        ),
        "feature_count": (
            len(selected_features)
        ),
        "features": ",".join(
            selected_features
        ),
        **parameter_record,
        "feature_chromosome": "".join(
            str(int(gene))
            for gene
            in individual.feature_mask
        ),
        "parameter_chromosome": ",".join(
            str(int(gene))
            for gene
            in individual.parameter_genes
        ),
    }


def save_ga_summary(
    output_dir: Path,
    generation_history: list[dict],
    individual_history: list[dict],
    best_individual: Individual,
    best_evaluation: EvaluationResult,
    all_features: list[str],
    evaluation_cache: dict[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
        ],
        EvaluationResult,
    ],
    fitness_lambda: float,
) -> Path:
    output_path = (
        output_dir
        / "ga_summary.xlsx"
    )

    generation_df = pd.DataFrame(
        generation_history
    )

    individual_df = pd.DataFrame(
        individual_history
    )

    best_features = (
        selected_features_from_mask(
            best_individual.feature_mask,
            all_features,
        )
    )

    best_features_df = pd.DataFrame({
        "feature": best_features,
    })

    best_parameters_df = pd.DataFrame([
        {
            "fitness": (
                best_evaluation.fitness
            ),
            "auc_mean": (
                best_evaluation.auc_mean
            ),
            "auc_std": (
                best_evaluation.auc_std
            ),
            "fitness_lambda": (
                fitness_lambda
            ),
            **parameter_genes_to_record(
                best_individual.parameter_genes
            ),
        }
    ])

    chromosome_df = pd.DataFrame({
        "feature": all_features,
        "selected": (
            best_individual
            .feature_mask
            .astype(int)
        ),
    })

    evaluated_rows: list[dict] = []

    for (
        feature_key,
        parameter_key,
    ), evaluation in (
        evaluation_cache.items()
    ):
        feature_mask = np.asarray(
            feature_key,
            dtype=np.int8,
        )

        parameter_genes = np.asarray(
            parameter_key,
            dtype=np.int16,
        )

        selected_features = (
            selected_features_from_mask(
                feature_mask,
                all_features,
            )
        )

        evaluated_rows.append({
            "fitness": (
                evaluation.fitness
            ),
            "auc_mean": (
                evaluation.auc_mean
            ),
            "auc_std": (
                evaluation.auc_std
            ),
            "feature_count": (
                len(selected_features)
            ),
            "features": ",".join(
                selected_features
            ),
            **parameter_genes_to_record(
                parameter_genes
            ),
            "feature_chromosome": (
                "".join(
                    str(int(gene))
                    for gene
                    in feature_mask
                )
            ),
            "parameter_chromosome": (
                ",".join(
                    str(int(gene))
                    for gene
                    in parameter_genes
                )
            ),
        })

    evaluated_df = pd.DataFrame(
        evaluated_rows
    )

    if not evaluated_df.empty:
        evaluated_df = (
            evaluated_df.sort_values(
                "fitness",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        generation_df.to_excel(
            writer,
            sheet_name=(
                "generation_summary"
            ),
            index=False,
        )

        individual_df.to_excel(
            writer,
            sheet_name=(
                "individual_history"
            ),
            index=False,
        )

        best_features_df.to_excel(
            writer,
            sheet_name="best_features",
            index=False,
        )

        best_parameters_df.to_excel(
            writer,
            sheet_name="best_parameters",
            index=False,
        )

        chromosome_df.to_excel(
            writer,
            sheet_name=(
                "best_feature_chromosome"
            ),
            index=False,
        )

        evaluated_df.to_excel(
            writer,
            sheet_name="evaluated_sets",
            index=False,
        )

    return output_path


def run_ga_on_outer_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    all_features: list[str],
    output_dir: Path,
    population_size: int,
    generations: int,
    crossover_rate: float,
    elite_size: int,
    tournament_size: int,
    min_features: int,
    max_features: int,
    feature_mutation_rate: float,
    parameter_mutation_rate: float,
    fitness_lambda: float,
    inner_n_splits: int,
    inner_n_repeats: int,
    random_seed: int,
    thread_count: int,
) -> tuple[
    Individual,
    EvaluationResult,
    Path,
]:
    random.seed(
        random_seed
    )
    np.random.seed(
        random_seed
    )

    rng = np.random.default_rng(
        random_seed
    )

    population = initialize_population(
        population_size=(
            population_size
        ),
        feature_count=(
            len(all_features)
        ),
        min_features=(
            min_features
        ),
        max_features=(
            max_features
        ),
        rng=rng,
    )

    evaluation_cache: dict[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
        ],
        EvaluationResult,
    ] = {}

    generation_history: list[dict] = []
    individual_history: list[dict] = []

    global_best_individual: (
        Individual | None
    ) = None

    global_best_evaluation = (
        EvaluationResult(
            fitness=float("-inf"),
            auc_mean=float("nan"),
            auc_std=float("nan"),
        )
    )

    for generation in range(
        1,
        generations + 1,
    ):
        print(
            "\n  "
            + "=" * 58
        )
        print(
            f"  Generation "
            f"{generation}/"
            f"{generations}"
        )
        print(
            "  "
            + "=" * 58
        )

        unique_individuals: dict[
            tuple[
                tuple[int, ...],
                tuple[int, ...],
            ],
            Individual,
        ] = {}

        for individual in population:
            unique_individuals.setdefault(
                individual.key(),
                individual,
            )

        keys_to_evaluate = [
            key
            for key
            in unique_individuals
            if key
            not in evaluation_cache
        ]

        print(
            "  個体数: "
            f"{len(population)} / "
            "ユニーク: "
            f"{len(unique_individuals)} / "
            "新規評価: "
            f"{len(keys_to_evaluate)}"
        )

        def evaluate_key(
            key: tuple[
                tuple[int, ...],
                tuple[int, ...],
            ],
        ) -> tuple[
            tuple[
                tuple[int, ...],
                tuple[int, ...],
            ],
            EvaluationResult,
        ]:
            individual = (
                unique_individuals[
                    key
                ]
            )

            evaluation = (
                evaluate_individual(
                    individual=(
                        individual
                    ),
                    all_features=(
                        all_features
                    ),
                    X_train=X_train,
                    y_train=y_train,
                    config=config,
                    inner_n_splits=(
                        inner_n_splits
                    ),
                    inner_n_repeats=(
                        inner_n_repeats
                    ),
                    fitness_lambda=(
                        fitness_lambda
                    ),
                    model_random_state=(
                        random_seed
                    ),
                )
            )

            return (
                key,
                evaluation,
            )

        if keys_to_evaluate:
            with ThreadPoolExecutor(
                max_workers=(
                    thread_count
                )
            ) as executor:
                for key, evaluation in (
                    executor.map(
                        evaluate_key,
                        keys_to_evaluate,
                    )
                ):
                    evaluation_cache[
                        key
                    ] = evaluation

        evaluations = [
            evaluation_cache[
                individual.key()
            ]
            for individual
            in population
        ]

        fitnesses = [
            evaluation.fitness
            for evaluation
            in evaluations
        ]

        finite_indexes = [
            index
            for index, fitness
            in enumerate(fitnesses)
            if np.isfinite(fitness)
        ]

        if not finite_indexes:
            raise RuntimeError(
                "全個体の評価に失敗しました。"
            )

        best_index = max(
            finite_indexes,
            key=lambda index: (
                fitnesses[index]
            ),
        )

        generation_best_individual = (
            population[
                best_index
            ].copy()
        )

        generation_best_evaluation = (
            evaluations[
                best_index
            ]
        )

        if (
            generation_best_evaluation
            .fitness
            > global_best_evaluation
            .fitness
        ):
            global_best_individual = (
                generation_best_individual
                .copy()
            )

            global_best_evaluation = (
                generation_best_evaluation
            )

        for index, (
            individual,
            evaluation,
        ) in enumerate(
            zip(
                population,
                evaluations,
            ),
            start=1,
        ):
            row = (
                individual_to_history_row(
                    individual=individual,
                    all_features=(
                        all_features
                    ),
                    evaluation=evaluation,
                    generation=generation,
                    individual_number=(
                        index
                    ),
                    fitness_lambda=(
                        fitness_lambda
                    ),
                )
            )

            individual_history.append(
                row
            )

            print(
                f"  Individual "
                f"{index:02d}: "
                f"AUC="
                f"{evaluation.auc_mean:.6f} "
                f"± "
                f"{evaluation.auc_std:.6f}, "
                f"Fitness="
                f"{evaluation.fitness:.6f}, "
                f"features="
                f"{row['feature_count']}, "
                f"pair="
                f"{row['estimator_lr_pair_no']}/"
                f"{row['depth_leaves_pair_no']}"
            )

        finite_fitnesses = np.asarray(
            [
                fitnesses[index]
                for index
                in finite_indexes
            ],
            dtype=float,
        )

        best_features = (
            selected_features_from_mask(
                generation_best_individual
                .feature_mask,
                all_features,
            )
        )

        best_params = (
            parameter_genes_to_record(
                generation_best_individual
                .parameter_genes
            )
        )

        generation_history.append({
            "generation": generation,
            "generation_best_fitness": (
                generation_best_evaluation
                .fitness
            ),
            "generation_best_auc_mean": (
                generation_best_evaluation
                .auc_mean
            ),
            "generation_best_auc_std": (
                generation_best_evaluation
                .auc_std
            ),
            "generation_fitness_mean": float(
                np.mean(
                    finite_fitnesses
                )
            ),
            "generation_fitness_std": float(
                np.std(
                    finite_fitnesses,
                    ddof=1,
                )
                if len(
                    finite_fitnesses
                ) >= 2
                else 0.0
            ),
            "global_best_fitness": (
                global_best_evaluation
                .fitness
            ),
            "global_best_auc_mean": (
                global_best_evaluation
                .auc_mean
            ),
            "global_best_auc_std": (
                global_best_evaluation
                .auc_std
            ),
            "best_feature_count": (
                len(best_features)
            ),
            "best_features": ",".join(
                best_features
            ),
            **best_params,
            "unique_evaluated_total": (
                len(
                    evaluation_cache
                )
            ),
        })

        print(
            "  世代最良: "
            f"AUC="
            f"{generation_best_evaluation.auc_mean:.6f} "
            f"± "
            f"{generation_best_evaluation.auc_std:.6f}, "
            f"Fitness="
            f"{generation_best_evaluation.fitness:.6f}"
        )

        print(
            "  全世代最良: "
            f"AUC="
            f"{global_best_evaluation.auc_mean:.6f} "
            f"± "
            f"{global_best_evaluation.auc_std:.6f}, "
            f"Fitness="
            f"{global_best_evaluation.fitness:.6f}"
        )

        # Early stopping is intentionally NOT used.
        if generation == generations:
            break

        sorted_indexes = sorted(
            range(
                len(population)
            ),
            key=lambda index: (
                fitnesses[index]
            ),
            reverse=True,
        )

        next_population: list[
            Individual
        ] = [
            population[
                index
            ].copy()
            for index
            in sorted_indexes[
                :elite_size
            ]
        ]

        while (
            len(next_population)
            < population_size
        ):
            parent1 = (
                tournament_selection(
                    population=(
                        population
                    ),
                    fitnesses=fitnesses,
                    tournament_size=(
                        tournament_size
                    ),
                    rng=rng,
                )
            )

            parent2 = (
                tournament_selection(
                    population=(
                        population
                    ),
                    fitnesses=fitnesses,
                    tournament_size=(
                        tournament_size
                    ),
                    rng=rng,
                )
            )

            child1, child2 = (
                uniform_crossover(
                    parent1=parent1,
                    parent2=parent2,
                    crossover_rate=(
                        crossover_rate
                    ),
                    rng=rng,
                )
            )

            child1 = mutate_individual(
                individual=child1,
                feature_mutation_rate=(
                    feature_mutation_rate
                ),
                parameter_mutation_rate=(
                    parameter_mutation_rate
                ),
                min_features=(
                    min_features
                ),
                max_features=(
                    max_features
                ),
                rng=rng,
            )

            child2 = mutate_individual(
                individual=child2,
                feature_mutation_rate=(
                    feature_mutation_rate
                ),
                parameter_mutation_rate=(
                    parameter_mutation_rate
                ),
                min_features=(
                    min_features
                ),
                max_features=(
                    max_features
                ),
                rng=rng,
            )

            next_population.append(
                child1
            )

            if (
                len(next_population)
                < population_size
            ):
                next_population.append(
                    child2
                )

        population = next_population

    if global_best_individual is None:
        raise RuntimeError(
            "最良個体を取得できませんでした。"
        )

    ga_summary_path = (
        save_ga_summary(
            output_dir=output_dir,
            generation_history=(
                generation_history
            ),
            individual_history=(
                individual_history
            ),
            best_individual=(
                global_best_individual
            ),
            best_evaluation=(
                global_best_evaluation
            ),
            all_features=(
                all_features
            ),
            evaluation_cache=(
                evaluation_cache
            ),
            fitness_lambda=(
                fitness_lambda
            ),
        )
    )

    return (
        global_best_individual,
        global_best_evaluation,
        ga_summary_path,
    )


def evaluate_outer_test_once(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_individual: Individual,
    all_features: list[str],
    test_indexes: np.ndarray,
    model_random_state: int,
) -> tuple[
    float,
    pd.DataFrame,
    pd.DataFrame,
]:
    best_features = (
        selected_features_from_mask(
            best_individual.feature_mask,
            all_features,
        )
    )

    model_params = (
        build_individual_model_params(
            individual=best_individual,
            model_random_state=(
                model_random_state
            ),
        )
    )

    evaluation = (
        evaluate_lightgbm_holdout(
            X_train_original=(
                X_train[
                    best_features
                ].copy()
            ),
            y_train=y_train,
            X_test_original=(
                X_test[
                    best_features
                ].copy()
            ),
            y_test=y_test,
            default_threshold=(
                DEFAULT_THRESHOLD
            ),
            model_params=model_params,
        )
    )

    predictions = evaluation[
        "predictions"
    ].copy()

    predictions.insert(
        0,
        "filtered_row",
        test_indexes + 1,
    )

    feature_importance = evaluation[
        "feature_importance"
    ].copy()

    feature_importance.insert(
        0,
        "rank",
        range(
            1,
            len(feature_importance)
            + 1,
        ),
    )

    return (
        float(
            evaluation["roc_auc"]
        ),
        predictions,
        feature_importance,
    )


def dataset_signature(
    X: pd.DataFrame,
    y: pd.Series,
) -> str:
    hasher = hashlib.sha256()

    x_hash = (
        pd.util.hash_pandas_object(
            X,
            index=True,
        )
        .to_numpy()
        .tobytes()
    )

    y_hash = (
        pd.util.hash_pandas_object(
            y,
            index=True,
        )
        .to_numpy()
        .tobytes()
    )

    hasher.update(
        x_hash
    )
    hasher.update(
        y_hash
    )

    return hasher.hexdigest()


def candidate_space_signature() -> str:
    payload = {
        "estimator_lr_pairs": (
            ESTIMATOR_LR_PAIRS
        ),
        "depth_leaves_pairs": (
            DEPTH_LEAVES_PAIRS
        ),
        "min_child_samples": (
            MIN_CHILD_SAMPLES_VALUES
        ),
        "subsample": (
            SUBSAMPLE_VALUES
        ),
        "colsample_bytree": (
            COLSAMPLE_BYTREE_VALUES
        ),
        "reg_alpha": (
            REG_ALPHA_VALUES
        ),
        "reg_lambda": (
            REG_LAMBDA_VALUES
        ),
    }

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(
        serialized.encode(
            "utf-8"
        )
    ).hexdigest()


def build_run_signature(
    *,
    dataset_hash: str,
    all_features: list[str],
    population_size: int,
    generations: int,
    crossover_rate: float,
    elite_size: int,
    tournament_size: int,
    min_features: int,
    max_features: int,
    feature_mutation_rate: float,
    parameter_mutation_rate: float,
    fitness_lambda: float,
    inner_n_splits: int,
    inner_n_repeats: int,
    outer_n_splits: int,
    outer_n_repeats: int,
    random_seed: int,
    outer_random_state: int,
) -> str:
    payload = {
        "dataset_hash": (
            dataset_hash
        ),
        "all_features": (
            all_features
        ),
        "population_size": (
            population_size
        ),
        "generations": (
            generations
        ),
        "crossover_rate": (
            crossover_rate
        ),
        "elite_size": (
            elite_size
        ),
        "tournament_size": (
            tournament_size
        ),
        "min_features": (
            min_features
        ),
        "max_features": (
            max_features
        ),
        "feature_mutation_rate": (
            feature_mutation_rate
        ),
        "parameter_mutation_rate": (
            parameter_mutation_rate
        ),
        "fitness_lambda": (
            fitness_lambda
        ),
        "inner_n_splits": (
            inner_n_splits
        ),
        "inner_n_repeats": (
            inner_n_repeats
        ),
        "outer_n_splits": (
            outer_n_splits
        ),
        "outer_n_repeats": (
            outer_n_repeats
        ),
        "random_seed": (
            random_seed
        ),
        "outer_random_state": (
            outer_random_state
        ),
        "candidate_space_signature": (
            candidate_space_signature()
        ),
    }

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(
        serialized.encode(
            "utf-8"
        )
    ).hexdigest()


def save_outer_fold_result(
    output_dir: Path,
    repeat_number: int,
    fold_number: int,
    split_number: int,
    train_indexes: np.ndarray,
    test_indexes: np.ndarray,
    best_individual: Individual,
    best_evaluation: EvaluationResult,
    all_features: list[str],
    outer_test_auc: float,
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    run_signature: str,
    population_size: int,
    generations: int,
    fitness_lambda: float,
    feature_mutation_rate: float,
    parameter_mutation_rate: float,
    inner_n_splits: int,
    inner_n_repeats: int,
) -> Path:
    output_path = (
        output_dir
        / "outer_test_result.xlsx"
    )

    best_features = (
        selected_features_from_mask(
            best_individual.feature_mask,
            all_features,
        )
    )

    parameter_record = (
        parameter_genes_to_record(
            best_individual.parameter_genes
        )
    )

    summary = pd.DataFrame([
        {
            "status": "complete",
            "outer_repeat": (
                repeat_number
            ),
            "outer_fold": (
                fold_number
            ),
            "outer_split": (
                split_number
            ),
            "train_count": (
                len(train_indexes)
            ),
            "test_count": (
                len(test_indexes)
            ),
            "ga_best_fitness": (
                best_evaluation.fitness
            ),
            "ga_best_auc_mean": (
                best_evaluation.auc_mean
            ),
            "ga_best_auc_std": (
                best_evaluation.auc_std
            ),
            "fitness_lambda": (
                fitness_lambda
            ),
            "outer_test_auc": (
                outer_test_auc
            ),
            "feature_count": (
                len(best_features)
            ),
            "population_size": (
                population_size
            ),
            "generations": (
                generations
            ),
            "feature_mutation_rate": (
                feature_mutation_rate
            ),
            "parameter_mutation_rate": (
                parameter_mutation_rate
            ),
            "inner_n_splits": (
                inner_n_splits
            ),
            "inner_n_repeats": (
                inner_n_repeats
            ),
            "parameter_combination_count": (
                PARAMETER_COMBINATION_COUNT
            ),
            "run_signature": (
                run_signature
            ),
            **parameter_record,
        }
    ])

    best_features_df = pd.DataFrame({
        "feature": best_features,
    })

    best_parameters_df = pd.DataFrame([
        {
            **parameter_record,
        }
    ])

    split_data = pd.concat([
        pd.DataFrame({
            "filtered_row": (
                train_indexes + 1
            ),
            "split": "train",
        }),
        pd.DataFrame({
            "filtered_row": (
                test_indexes + 1
            ),
            "split": "test",
        }),
    ], ignore_index=True).sort_values(
        "filtered_row"
    ).reset_index(
        drop=True
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        summary.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        best_features_df.to_excel(
            writer,
            sheet_name="best_features",
            index=False,
        )

        best_parameters_df.to_excel(
            writer,
            sheet_name="best_parameters",
            index=False,
        )

        predictions.to_excel(
            writer,
            sheet_name="test_predictions",
            index=False,
        )

        feature_importance.to_excel(
            writer,
            sheet_name="feature_importance",
            index=False,
        )

        split_data.to_excel(
            writer,
            sheet_name="data_split",
            index=False,
        )

    return output_path


def is_outer_fold_complete(
    output_dir: Path,
    repeat_number: int,
    fold_number: int,
    run_signature: str,
) -> tuple[bool, str]:
    result_path = (
        output_dir
        / "outer_test_result.xlsx"
    )

    if not result_path.exists():
        return (
            False,
            "outer_test_result.xlsx "
            "が存在しません",
        )

    try:
        summary = pd.read_excel(
            result_path,
            sheet_name="summary",
        )

        required_columns = {
            "status",
            "outer_repeat",
            "outer_fold",
            "outer_test_auc",
            "run_signature",
        }

        missing_columns = (
            required_columns
            - set(
                summary.columns
            )
        )

        if missing_columns:
            return (
                False,
                "summaryの必要列不足: "
                + ", ".join(
                    sorted(
                        missing_columns
                    )
                ),
            )

        if summary.empty:
            return (
                False,
                "summaryが空です",
            )

        row = summary.iloc[0]

        if (
            str(row["status"])
            != "complete"
        ):
            return (
                False,
                "statusがcompleteでは"
                "ありません",
            )

        if (
            int(row["outer_repeat"])
            != repeat_number
            or int(row["outer_fold"])
            != fold_number
        ):
            return (
                False,
                "repeat/foldが"
                "一致しません",
            )

        if (
            str(row["run_signature"])
            != run_signature
        ):
            return (
                False,
                "実験条件が以前の"
                "結果と異なります",
            )

        outer_test_auc = float(
            row["outer_test_auc"]
        )

        if not np.isfinite(
            outer_test_auc
        ):
            return (
                False,
                "outer_test_aucが"
                "有限値ではありません",
            )

    except Exception as exc:
        return (
            False,
            "既存結果を"
            "読み込めません: "
            f"{exc}",
        )

    return (
        True,
        "完了済み結果を確認",
    )


def run_genetic_algorithm(
    config_path: str | Path,
    data_path: str | Path | None = None,
    population_size: int = POPULATION_SIZE,
    generations: int = GENERATIONS,
    crossover_rate: float = CROSSOVER_RATE,
    elite_size: int = ELITE_SIZE,
    tournament_size: int = TOURNAMENT_SIZE,
    min_features: int = 1,
    max_features: int | None = None,
    feature_mutation_rate: float | None = None,
    parameter_mutation_rate: float | None = None,
    fitness_lambda: float = FITNESS_LAMBDA,
    inner_n_splits: int = INNER_N_SPLITS,
    inner_n_repeats: int = INNER_N_REPEATS,
    outer_n_splits: int = OUTER_N_SPLITS,
    outer_n_repeats: int = OUTER_N_REPEATS,
    random_seed: int = RANDOM_SEED,
    outer_random_state: int = OUTER_RANDOM_STATE,
    thread_count: int = GA_THREAD_COUNT,
    output_path: str | Path | None = None,
) -> Path:
    if data_path is None:
        (
            X_all,
            y_all,
            config,
        ) = filter_data(
            config_path
        )
    else:
        (
            X_all,
            y_all,
            config,
        ) = filter_data(
            config_path,
            data_path,
        )

    if y_all.nunique() != 2:
        raise ValueError(
            "このプログラムは"
            "二値分類専用です。"
        )

    all_features = (
        X_all.columns
        .astype(str)
        .tolist()
    )

    feature_count = len(
        all_features
    )

    if feature_count == 0:
        raise ValueError(
            "特徴量がありません。"
        )

    if max_features is None:
        max_features = (
            feature_count
        )

    if not (
        1
        <= min_features
        <= max_features
        <= feature_count
    ):
        raise ValueError(
            "1 <= min_features "
            "<= max_features "
            "<= feature_count "
            "としてください。"
        )

    if population_size < 2:
        raise ValueError(
            "population_sizeは"
            "2以上にしてください。"
        )

    if generations < 1:
        raise ValueError(
            "generationsは"
            "1以上にしてください。"
        )

    if not (
        0
        <= elite_size
        < population_size
    ):
        raise ValueError(
            "elite_sizeは"
            "0以上かつpopulation_size"
            "未満にしてください。"
        )

    if tournament_size < 2:
        raise ValueError(
            "tournament_sizeは"
            "2以上にしてください。"
        )

    if not (
        0.0
        <= crossover_rate
        <= 1.0
    ):
        raise ValueError(
            "crossover_rateは"
            "0～1にしてください。"
        )

    if fitness_lambda < 0.0:
        raise ValueError(
            "fitness_lambdaは"
            "0以上にしてください。"
        )

    if inner_n_splits < 2:
        raise ValueError(
            "inner_n_splitsは"
            "2以上にしてください。"
        )

    if inner_n_repeats < 1:
        raise ValueError(
            "inner_n_repeatsは"
            "1以上にしてください。"
        )

    if outer_n_splits < 2:
        raise ValueError(
            "outer_n_splitsは"
            "2以上にしてください。"
        )

    if outer_n_repeats < 1:
        raise ValueError(
            "outer_n_repeatsは"
            "1以上にしてください。"
        )

    if thread_count < 1:
        raise ValueError(
            "threadsは"
            "1以上にしてください。"
        )

    if (
        feature_mutation_rate
        is None
    ):
        feature_mutation_rate = (
            1.0
            / feature_count
        )

    if (
        parameter_mutation_rate
        is None
    ):
        parameter_mutation_rate = (
            1.0
            / PARAMETER_GENE_COUNT
        )

    if not (
        0.0
        <= feature_mutation_rate
        <= 1.0
    ):
        raise ValueError(
            "feature_mutation_rateは"
            "0～1にしてください。"
        )

    if not (
        0.0
        <= parameter_mutation_rate
        <= 1.0
    ):
        raise ValueError(
            "parameter_mutation_rateは"
            "0～1にしてください。"
        )

    experiment_name = config[
        "experiment_name"
    ]

    if output_path is None:
        output_dir = (
            RESULT_DIR
            / (
                "GA_FeatureHyper_"
                + str(
                    experiment_name
                )
            )
        )
    else:
        output_dir = Path(
            output_path
        )

        if not (
            output_dir.is_absolute()
        ):
            output_dir = (
                BASE_DIR
                / output_dir
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_hash = dataset_signature(
        X_all,
        y_all,
    )

    run_signature = (
        build_run_signature(
            dataset_hash=(
                data_hash
            ),
            all_features=(
                all_features
            ),
            population_size=(
                population_size
            ),
            generations=(
                generations
            ),
            crossover_rate=(
                crossover_rate
            ),
            elite_size=(
                elite_size
            ),
            tournament_size=(
                tournament_size
            ),
            min_features=(
                min_features
            ),
            max_features=(
                max_features
            ),
            feature_mutation_rate=(
                feature_mutation_rate
            ),
            parameter_mutation_rate=(
                parameter_mutation_rate
            ),
            fitness_lambda=(
                fitness_lambda
            ),
            inner_n_splits=(
                inner_n_splits
            ),
            inner_n_repeats=(
                inner_n_repeats
            ),
            outer_n_splits=(
                outer_n_splits
            ),
            outer_n_repeats=(
                outer_n_repeats
            ),
            random_seed=(
                random_seed
            ),
            outer_random_state=(
                outer_random_state
            ),
        )
    )

    print(
        "=" * 70
    )
    print(
        "GA Feature + "
        "Hyperparameter Search"
    )
    print(
        "=" * 70
    )
    print(
        f"実験名: "
        f"{experiment_name}"
    )
    print(
        f"被験者数: "
        f"{len(X_all)}"
    )
    print(
        f"特徴量数: "
        f"{feature_count}"
    )
    print(
        "パラメータ候補: "
        f"{PARAMETER_COMBINATION_COUNT:,}"
        "通り"
    )
    print(
        "外側CV: "
        f"{outer_n_splits}-fold × "
        f"{outer_n_repeats}反復"
    )
    print(
        "各個体の内部CV: "
        f"{inner_n_splits}-fold × "
        f"{inner_n_repeats}反復"
    )
    print(
        f"個体数: "
        f"{population_size}"
    )
    print(
        f"世代数: "
        f"{generations}"
    )
    print(
        "早期終了: なし"
    )
    print(
        "Fitness = "
        "mean(AUC) - "
        f"{fitness_lambda:g}"
        " × std(AUC)"
    )
    print(
        "特徴量突然変異率: "
        f"{feature_mutation_rate:.8f} "
        "(期待変異数≈1)"
    )
    print(
        "パラメータ突然変異率: "
        f"{parameter_mutation_rate:.8f} "
        "(期待変異数≈1)"
    )
    print(
        "パラメータ変異: "
        "隣接候補のみ"
    )
    print(
        "LightGBM評価: "
        "run_lightgbm.py の "
        "evaluate_lightgbm / "
        "evaluate_lightgbm_holdout を使用"
    )
    print(
        f"交叉率: "
        f"{crossover_rate}"
    )
    print(
        f"エリート数: "
        f"{elite_size}"
    )
    print(
        f"トーナメントサイズ: "
        f"{tournament_size}"
    )
    print(
        f"並列スレッド数: "
        f"{thread_count}"
    )
    print(
        f"出力先: "
        f"{output_dir}"
    )

    outer_cv = (
        RepeatedStratifiedKFold(
            n_splits=(
                outer_n_splits
            ),
            n_repeats=(
                outer_n_repeats
            ),
            random_state=(
                outer_random_state
            ),
        )
    )

    total_outer_splits = (
        outer_n_splits
        * outer_n_repeats
    )

    executed_count = 0
    skipped_count = 0

    for split_number, (
        train_indexes,
        test_indexes,
    ) in enumerate(
        outer_cv.split(
            X_all,
            y_all,
        ),
        start=1,
    ):
        (
            repeat_number,
            fold_number,
        ) = get_repeat_and_fold(
            split_number,
            outer_n_splits,
        )

        print(
            "\n"
            + "=" * 70
        )
        print(
            f"[{current_timestamp()}] "
            f"OUTER Repeat "
            f"{repeat_number}/"
            f"{outer_n_repeats}, "
            f"Fold "
            f"{fold_number}/"
            f"{outer_n_splits} "
            f"({split_number}/"
            f"{total_outer_splits})"
        )
        print(
            "=" * 70
        )

        fold_output_dir = (
            output_dir
            / (
                f"repeat_"
                f"{repeat_number:02d}"
            )
            / (
                f"fold_"
                f"{fold_number:02d}"
            )
        )

        (
            complete,
            reason,
        ) = is_outer_fold_complete(
            output_dir=(
                fold_output_dir
            ),
            repeat_number=(
                repeat_number
            ),
            fold_number=(
                fold_number
            ),
            run_signature=(
                run_signature
            ),
        )

        if complete:
            skipped_count += 1
            print(
                "  完了済みのため"
                "スキップ: "
                f"{reason}"
            )
            continue

        if fold_output_dir.exists():
            print(
                "  既存フォルダは"
                "ありますが再実行します: "
                f"{reason}"
            )

        fold_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        X_train = (
            X_all.iloc[
                train_indexes
            ]
            .reset_index(
                drop=True
            )
        )

        X_test = (
            X_all.iloc[
                test_indexes
            ]
            .reset_index(
                drop=True
            )
        )

        y_train = (
            y_all.iloc[
                train_indexes
            ]
            .reset_index(
                drop=True
            )
        )

        y_test = (
            y_all.iloc[
                test_indexes
            ]
            .reset_index(
                drop=True
            )
        )

        fold_ga_seed = (
            random_seed
            + split_number
            - 1
        )

        (
            best_individual,
            best_evaluation,
            ga_summary_path,
        ) = run_ga_on_outer_train(
            X_train=X_train,
            y_train=y_train,
            config=config,
            all_features=(
                all_features
            ),
            output_dir=(
                fold_output_dir
            ),
            population_size=(
                population_size
            ),
            generations=(
                generations
            ),
            crossover_rate=(
                crossover_rate
            ),
            elite_size=(
                elite_size
            ),
            tournament_size=(
                tournament_size
            ),
            min_features=(
                min_features
            ),
            max_features=(
                max_features
            ),
            feature_mutation_rate=(
                feature_mutation_rate
            ),
            parameter_mutation_rate=(
                parameter_mutation_rate
            ),
            fitness_lambda=(
                fitness_lambda
            ),
            inner_n_splits=(
                inner_n_splits
            ),
            inner_n_repeats=(
                inner_n_repeats
            ),
            random_seed=(
                fold_ga_seed
            ),
            thread_count=(
                thread_count
            ),
        )

        (
            outer_test_auc,
            predictions,
            feature_importance,
        ) = evaluate_outer_test_once(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            best_individual=(
                best_individual
            ),
            all_features=(
                all_features
            ),
            test_indexes=(
                test_indexes
            ),
            model_random_state=(
                fold_ga_seed
                + 1_000_000
            ),
        )

        predictions.insert(
            0,
            "outer_split",
            split_number,
        )

        predictions.insert(
            0,
            "outer_fold",
            fold_number,
        )

        predictions.insert(
            0,
            "outer_repeat",
            repeat_number,
        )

        result_path = (
            save_outer_fold_result(
                output_dir=(
                    fold_output_dir
                ),
                repeat_number=(
                    repeat_number
                ),
                fold_number=(
                    fold_number
                ),
                split_number=(
                    split_number
                ),
                train_indexes=(
                    train_indexes
                ),
                test_indexes=(
                    test_indexes
                ),
                best_individual=(
                    best_individual
                ),
                best_evaluation=(
                    best_evaluation
                ),
                all_features=(
                    all_features
                ),
                outer_test_auc=(
                    outer_test_auc
                ),
                predictions=(
                    predictions
                ),
                feature_importance=(
                    feature_importance
                ),
                run_signature=(
                    run_signature
                ),
                population_size=(
                    population_size
                ),
                generations=(
                    generations
                ),
                fitness_lambda=(
                    fitness_lambda
                ),
                feature_mutation_rate=(
                    feature_mutation_rate
                ),
                parameter_mutation_rate=(
                    parameter_mutation_rate
                ),
                inner_n_splits=(
                    inner_n_splits
                ),
                inner_n_repeats=(
                    inner_n_repeats
                ),
            )
        )

        (
            saved_ok,
            saved_reason,
        ) = is_outer_fold_complete(
            output_dir=(
                fold_output_dir
            ),
            repeat_number=(
                repeat_number
            ),
            fold_number=(
                fold_number
            ),
            run_signature=(
                run_signature
            ),
        )

        if not saved_ok:
            raise RuntimeError(
                "保存後検証に"
                "失敗しました: "
                f"{saved_reason}"
            )

        executed_count += 1

        best_features = (
            selected_features_from_mask(
                best_individual
                .feature_mask,
                all_features,
            )
        )

        best_parameters = (
            parameter_genes_to_record(
                best_individual
                .parameter_genes
            )
        )

        print(
            "\n  ----- "
            "outer TEST評価 -----"
        )
        print(
            "  GA内部CV AUC: "
            f"{best_evaluation.auc_mean:.6f} "
            f"± "
            f"{best_evaluation.auc_std:.6f}"
        )
        print(
            "  GA Fitness: "
            f"{best_evaluation.fitness:.6f}"
        )
        print(
            "  outer TEST AUC: "
            f"{outer_test_auc:.6f}"
        )
        print(
            "  選択特徴量数: "
            f"{len(best_features)}"
        )
        print(
            "  n_estimators / "
            "learning_rate: "
            f"{best_parameters['n_estimators']} / "
            f"{best_parameters['learning_rate']}"
        )
        print(
            "  max_depth / "
            "num_leaves: "
            f"{best_parameters['max_depth']} / "
            f"{best_parameters['num_leaves']}"
        )
        print(
            f"  GA履歴: "
            f"{ga_summary_path}"
        )
        print(
            f"  outer結果: "
            f"{result_path}"
        )

    print(
        "\n"
        + "=" * 70
    )
    print(
        "===== 実行終了 ====="
    )
    print(
        "=" * 70
    )
    print(
        "今回実行したouter fold: "
        f"{executed_count}"
    )
    print(
        "完了済みでスキップ: "
        f"{skipped_count}"
    )
    print(
        "全体集計ファイルは"
        "このプログラムでは"
        "作成しません。"
    )
    print(
        f"結果保存先: "
        f"{output_dir}"
    )

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LightGBMの特徴量と"
            "ハイパーパラメータを"
            "同時最適化するGA"
        )
    )

    parser.add_argument(
        "config",
        help="設定JSONのパス",
    )

    parser.add_argument(
        "--data",
        default=None,
        help="元データExcelのパス",
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=POPULATION_SIZE,
        help=(
            "個体数 "
            f"(default: "
            f"{POPULATION_SIZE})"
        ),
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=GENERATIONS,
        help=(
            "世代数。"
            "早期終了は使用しない "
            f"(default: "
            f"{GENERATIONS})"
        ),
    )

    parser.add_argument(
        "--fitness-lambda",
        type=float,
        default=FITNESS_LAMBDA,
        help=(
            "Fitness = "
            "mean(AUC) - lambda*std(AUC) "
            f"のlambda "
            f"(default: "
            f"{FITNESS_LAMBDA})"
        ),
    )

    parser.add_argument(
        "--inner-splits",
        type=int,
        default=INNER_N_SPLITS,
        help=(
            "各個体評価のCV分割数 "
            f"(default: "
            f"{INNER_N_SPLITS})"
        ),
    )

    parser.add_argument(
        "--inner-repeats",
        type=int,
        default=INNER_N_REPEATS,
        help=(
            "各個体評価のCV反復数 "
            f"(default: "
            f"{INNER_N_REPEATS})"
        ),
    )

    parser.add_argument(
        "--outer-splits",
        type=int,
        default=OUTER_N_SPLITS,
        help=(
            "外側CV分割数 "
            f"(default: "
            f"{OUTER_N_SPLITS})"
        ),
    )

    parser.add_argument(
        "--outer-repeats",
        type=int,
        default=OUTER_N_REPEATS,
        help=(
            "外側CV反復数 "
            f"(default: "
            f"{OUTER_N_REPEATS})"
        ),
    )

    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=CROSSOVER_RATE,
        help=(
            "交叉率 "
            f"(default: "
            f"{CROSSOVER_RATE})"
        ),
    )

    parser.add_argument(
        "--elite-size",
        type=int,
        default=ELITE_SIZE,
        help=(
            "エリート保存数 "
            f"(default: "
            f"{ELITE_SIZE})"
        ),
    )

    parser.add_argument(
        "--tournament-size",
        type=int,
        default=TOURNAMENT_SIZE,
        help=(
            "トーナメントサイズ "
            f"(default: "
            f"{TOURNAMENT_SIZE})"
        ),
    )

    parser.add_argument(
        "--min-features",
        type=int,
        default=1,
        help="最小特徴量数",
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help=(
            "最大特徴量数。"
            "省略時は全特徴量数"
        ),
    )

    parser.add_argument(
        "--feature-mutation-rate",
        type=float,
        default=None,
        help=(
            "各特徴量遺伝子の"
            "突然変異率。"
            "省略時は1/特徴量数"
            "（期待変異数1）"
        ),
    )

    parser.add_argument(
        "--parameter-mutation-rate",
        type=float,
        default=None,
        help=(
            "各パラメータ遺伝子の"
            "突然変異率。"
            "省略時は1/7"
            "（期待変異数1）。"
            "変異先は隣接候補"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=(
            "GA基準乱数seed "
            f"(default: "
            f"{RANDOM_SEED})"
        ),
    )

    parser.add_argument(
        "--outer-seed",
        type=int,
        default=OUTER_RANDOM_STATE,
        help=(
            "外側CV乱数seed "
            f"(default: "
            f"{OUTER_RANDOM_STATE})"
        ),
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=GA_THREAD_COUNT,
        help=(
            "個体並列評価の"
            "最大スレッド数 "
            f"(default: "
            f"{GA_THREAD_COUNT})"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="結果保存ディレクトリ",
    )

    args = parser.parse_args()

    run_genetic_algorithm(
        config_path=args.config,
        data_path=args.data,
        population_size=(
            args.population_size
        ),
        generations=args.generations,
        crossover_rate=(
            args.crossover_rate
        ),
        elite_size=args.elite_size,
        tournament_size=(
            args.tournament_size
        ),
        min_features=(
            args.min_features
        ),
        max_features=(
            args.max_features
        ),
        feature_mutation_rate=(
            args.feature_mutation_rate
        ),
        parameter_mutation_rate=(
            args.parameter_mutation_rate
        ),
        fitness_lambda=(
            args.fitness_lambda
        ),
        inner_n_splits=(
            args.inner_splits
        ),
        inner_n_repeats=(
            args.inner_repeats
        ),
        outer_n_splits=(
            args.outer_splits
        ),
        outer_n_repeats=(
            args.outer_repeats
        ),
        random_seed=args.seed,
        outer_random_state=(
            args.outer_seed
        ),
        thread_count=(
            args.threads
        ),
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
