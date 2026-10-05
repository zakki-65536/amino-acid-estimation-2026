from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
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
    statistic: str,
) -> float:
    row = summary_metrics[
        summary_metrics["metric"] == metric_name
    ]

    if row.empty:
        raise ValueError(
            f"summary_metrics に metric={metric_name} がありません。"
        )

    return float(row.iloc[0][statistic])


def chromosome_to_features(
    chromosome: np.ndarray,
    all_features: list[str],
) -> list[str]:
    return [
        feature
        for gene, feature in zip(chromosome, all_features)
        if gene == 1
    ]


def repair_chromosome(
    chromosome: np.ndarray,
    min_features: int,
    max_features: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    選択特徴量数が min_features ～ max_features の範囲に入るよう修正する。
    """
    chromosome = chromosome.copy()

    selected = np.flatnonzero(chromosome == 1)
    unselected = np.flatnonzero(chromosome == 0)

    if len(selected) < min_features:
        add_count = min_features - len(selected)

        add_indexes = rng.choice(
            unselected,
            size=add_count,
            replace=False,
        )

        chromosome[add_indexes] = 1

    selected = np.flatnonzero(chromosome == 1)

    if len(selected) > max_features:
        remove_count = len(selected) - max_features

        remove_indexes = rng.choice(
            selected,
            size=remove_count,
            replace=False,
        )

        chromosome[remove_indexes] = 0

    return chromosome


def initialize_population(
    population_size: int,
    feature_count: int,
    min_features: int,
    max_features: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    population: list[np.ndarray] = []

    # 最初の1個体には全特徴量を入れる。
    full = np.ones(feature_count, dtype=np.int8)
    population.append(
        repair_chromosome(
            full,
            min_features=min_features,
            max_features=max_features,
            rng=rng,
        )
    )

    while len(population) < population_size:
        selected_count = int(
            rng.integers(
                min_features,
                max_features + 1,
            )
        )

        chromosome = np.zeros(
            feature_count,
            dtype=np.int8,
        )

        selected_indexes = rng.choice(
            feature_count,
            size=selected_count,
            replace=False,
        )

        chromosome[selected_indexes] = 1
        population.append(chromosome)

    return population


def evaluate_chromosome(
    chromosome: np.ndarray,
    all_features: list[str],
    X_all: pd.DataFrame,
    y: pd.Series,
    config: dict,
    fitness_cache: dict[tuple[int, ...], float],
) -> float:
    """
    chromosome が表す特徴量集合で LightGBM を評価し、
    roc_auc の平均値を Fitness として返す。

    同じ染色体はキャッシュから返し、再学習を避ける。
    """
    key = tuple(int(x) for x in chromosome)

    if key in fitness_cache:
        return fitness_cache[key]

    selected_features = chromosome_to_features(
        chromosome,
        all_features,
    )

    if not selected_features:
        fitness_cache[key] = float("-inf")
        return fitness_cache[key]

    current_config = copy.deepcopy(config)
    current_config["features"] = selected_features

    X_current = X_all[
        selected_features
    ].copy()

    try:
        evaluation = evaluate_lightgbm(
            X_original=X_current,
            y=y,
            config=current_config,
        )

        summary_metrics = evaluation[
            "result_sheets"
        ]["summary_metrics"]

        auc_mean = get_metric(
            summary_metrics,
            "roc_auc",
            "mean",
        )

        if not np.isfinite(auc_mean):
            auc_mean = float("-inf")

    except Exception as exc:
        print(
            "評価失敗: "
            f"{len(selected_features)}項目 / {exc}"
        )
        auc_mean = float("-inf")

    fitness_cache[key] = auc_mean
    return auc_mean


def tournament_selection(
    population: list[np.ndarray],
    fitnesses: list[float],
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    tournament_size = min(
        tournament_size,
        len(population),
    )

    indexes = rng.choice(
        len(population),
        size=tournament_size,
        replace=False,
    )

    best_index = max(
        indexes,
        key=lambda i: fitnesses[int(i)],
    )

    return population[int(best_index)].copy()


def uniform_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() >= crossover_rate:
        return parent1.copy(), parent2.copy()

    mask = rng.random(len(parent1)) < 0.5

    child1 = np.where(
        mask,
        parent1,
        parent2,
    ).astype(np.int8)

    child2 = np.where(
        mask,
        parent2,
        parent1,
    ).astype(np.int8)

    return child1, child2


def mutate(
    chromosome: np.ndarray,
    mutation_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    chromosome = chromosome.copy()

    mutation_mask = (
        rng.random(len(chromosome))
        < mutation_rate
    )

    chromosome[mutation_mask] = (
        1 - chromosome[mutation_mask]
    )

    return chromosome


def save_ga_summary(
    output_dir: Path,
    generation_history: list[dict],
    individual_history: list[dict],
    best_features: list[str],
    all_features: list[str],
    best_chromosome: np.ndarray,
    fitness_cache: dict[tuple[int, ...], float],
) -> Path:
    summary_path = output_dir / "ga_summary.xlsx"

    generation_df = pd.DataFrame(
        generation_history
    )

    individuals_df = pd.DataFrame(
        individual_history
    )

    best_features_df = pd.DataFrame({
        "feature": best_features,
    })

    chromosome_df = pd.DataFrame({
        "feature": all_features,
        "selected": best_chromosome.astype(int),
    })

    cache_rows = []

    for chromosome_key, fitness in fitness_cache.items():
        selected_features = [
            feature
            for gene, feature in zip(
                chromosome_key,
                all_features,
            )
            if gene == 1
        ]

        cache_rows.append({
            "fitness_auc": fitness,
            "feature_count": len(selected_features),
            "features": ",".join(selected_features),
            "chromosome": "".join(
                str(gene)
                for gene in chromosome_key
            ),
        })

    evaluated_df = (
        pd.DataFrame(cache_rows)
        .sort_values(
            "fitness_auc",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    with pd.ExcelWriter(
        summary_path,
        engine="openpyxl",
    ) as writer:
        generation_df.to_excel(
            writer,
            sheet_name="generation_summary",
            index=False,
        )

        individuals_df.to_excel(
            writer,
            sheet_name="individual_history",
            index=False,
        )

        best_features_df.to_excel(
            writer,
            sheet_name="best_features",
            index=False,
        )

        chromosome_df.to_excel(
            writer,
            sheet_name="best_chromosome",
            index=False,
        )

        evaluated_df.to_excel(
            writer,
            sheet_name="evaluated_sets",
            index=False,
        )

    return summary_path


def run_genetic_algorithm(
    config_path: str | Path,
    data_path: str | Path | None = None,
    population_size: int = 20,
    generations: int = 20,
    crossover_rate: float = 0.8,
    mutation_rate: float | None = None,
    elite_size: int = 2,
    tournament_size: int = 3,
    min_features: int = 1,
    max_features: int | None = None,
    random_seed: int = 42,
    output_path: str | Path | None = None,
) -> Path:
    if data_path is None:
        X_all, y, config = filter_data(
            config_path
        )
    else:
        X_all, y, config = filter_data(
            config_path,
            data_path,
        )

    all_features = (
        X_all.columns
        .astype(str)
        .tolist()
    )

    feature_count = len(all_features)

    if feature_count == 0:
        raise ValueError(
            "特徴量が1つもありません。"
        )

    if max_features is None:
        max_features = feature_count

    if not (
        1 <= min_features
        <= max_features
        <= feature_count
    ):
        raise ValueError(
            "特徴量数の条件は "
            "1 <= min_features <= max_features "
            "<= 全特徴量数 としてください。"
        )

    if population_size < 2:
        raise ValueError(
            "population_size は2以上にしてください。"
        )

    if not (
        0 <= elite_size
        < population_size
    ):
        raise ValueError(
            "elite_size は 0以上かつ "
            "population_size 未満にしてください。"
        )

    if generations < 1:
        raise ValueError(
            "generations は1以上にしてください。"
        )

    if mutation_rate is None:
        # 特徴量数に応じたGAでよく使われる初期値。
        mutation_rate = 1.0 / feature_count

    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError(
            "mutation_rate は0～1にしてください。"
        )

    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError(
            "crossover_rate は0～1にしてください。"
        )

    random.seed(random_seed)
    np.random.seed(random_seed)
    rng = np.random.default_rng(
        random_seed
    )

    experiment_name = config[
        "experiment_name"
    ]

    if output_path is None:
        output_dir = (
            RESULT_DIR
            / f"GA_{experiment_name}"
        )
    else:
        output_dir = Path(output_path)

        if not output_dir.is_absolute():
            output_dir = (
                BASE_DIR / output_dir
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"実験名: {experiment_name}"
    )
    print(
        f"被験者数: {len(X_all)}"
    )
    print(
        f"全特徴量数: {feature_count}"
    )
    print(
        f"個体数: {population_size}"
    )
    print(
        f"世代数: {generations}"
    )
    print(
        f"交叉率: {crossover_rate}"
    )
    print(
        f"突然変異率: {mutation_rate}"
    )
    print(
        f"エリート数: {elite_size}"
    )
    print(
        "特徴量数制約: "
        f"{min_features}～{max_features}"
    )

    population = initialize_population(
        population_size=population_size,
        feature_count=feature_count,
        min_features=min_features,
        max_features=max_features,
        rng=rng,
    )

    fitness_cache: dict[
        tuple[int, ...],
        float
    ] = {}

    generation_history: list[dict] = []
    individual_history: list[dict] = []

    global_best_chromosome: (
        np.ndarray | None
    ) = None

    global_best_fitness = float("-inf")

    for generation in range(
        1,
        generations + 1,
    ):
        print(
            f"\n===== Generation "
            f"{generation}/{generations} ====="
        )

        fitnesses: list[float] = []

        for individual_index, chromosome in enumerate(
            population,
            start=1,
        ):
            fitness = evaluate_chromosome(
                chromosome=chromosome,
                all_features=all_features,
                X_all=X_all,
                y=y,
                config=config,
                fitness_cache=fitness_cache,
            )

            fitnesses.append(fitness)

            selected_features = (
                chromosome_to_features(
                    chromosome,
                    all_features,
                )
            )

            individual_history.append({
                "generation": generation,
                "individual": individual_index,
                "fitness_auc": fitness,
                "feature_count": len(
                    selected_features
                ),
                "features": ",".join(
                    selected_features
                ),
                "chromosome": "".join(
                    str(int(gene))
                    for gene in chromosome
                ),
            })

            print(
                f"Individual "
                f"{individual_index:02d}: "
                f"AUC={fitness:.6f}, "
                f"features="
                f"{len(selected_features)}"
            )

        finite_fitnesses = [
            value
            for value in fitnesses
            if np.isfinite(value)
        ]

        if not finite_fitnesses:
            raise RuntimeError(
                "全個体の評価に失敗しました。"
            )

        best_index = int(
            np.argmax(fitnesses)
        )

        generation_best_fitness = float(
            fitnesses[best_index]
        )

        generation_best_chromosome = (
            population[best_index].copy()
        )

        generation_best_features = (
            chromosome_to_features(
                generation_best_chromosome,
                all_features,
            )
        )

        generation_mean = float(
            np.mean(finite_fitnesses)
        )

        generation_std = float(
            np.std(
                finite_fitnesses,
                ddof=0,
            )
        )

        if (
            generation_best_fitness
            > global_best_fitness
        ):
            global_best_fitness = (
                generation_best_fitness
            )

            global_best_chromosome = (
                generation_best_chromosome.copy()
            )

        generation_history.append({
            "generation": generation,
            "best_auc": generation_best_fitness,
            "mean_auc": generation_mean,
            "std_auc": generation_std,
            "best_feature_count": len(
                generation_best_features
            ),
            "best_features": ",".join(
                generation_best_features
            ),
            "unique_evaluated_total": len(
                fitness_cache
            ),
            "global_best_auc": (
                global_best_fitness
            ),
        })

        print(
            "世代最良 AUC: "
            f"{generation_best_fitness:.6f}"
        )
        print(
            "世代平均 AUC: "
            f"{generation_mean:.6f} "
            f"± {generation_std:.6f}"
        )
        print(
            "世代最良特徴量数: "
            f"{len(generation_best_features)}"
        )
        print(
            "全世代最良 AUC: "
            f"{global_best_fitness:.6f}"
        )
        print(
            "評価済みユニーク個体数: "
            f"{len(fitness_cache)}"
        )

        if generation == generations:
            break

        sorted_indexes = sorted(
            range(len(population)),
            key=lambda i: fitnesses[i],
            reverse=True,
        )

        next_population: list[
            np.ndarray
        ] = [
            population[i].copy()
            for i in sorted_indexes[
                :elite_size
            ]
        ]

        while (
            len(next_population)
            < population_size
        ):
            parent1 = tournament_selection(
                population=population,
                fitnesses=fitnesses,
                tournament_size=tournament_size,
                rng=rng,
            )

            parent2 = tournament_selection(
                population=population,
                fitnesses=fitnesses,
                tournament_size=tournament_size,
                rng=rng,
            )

            child1, child2 = (
                uniform_crossover(
                    parent1=parent1,
                    parent2=parent2,
                    crossover_rate=crossover_rate,
                    rng=rng,
                )
            )

            child1 = mutate(
                child1,
                mutation_rate,
                rng,
            )

            child2 = mutate(
                child2,
                mutation_rate,
                rng,
            )

            child1 = repair_chromosome(
                child1,
                min_features=min_features,
                max_features=max_features,
                rng=rng,
            )

            child2 = repair_chromosome(
                child2,
                min_features=min_features,
                max_features=max_features,
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

    if global_best_chromosome is None:
        raise RuntimeError(
            "最良個体を取得できませんでした。"
        )

    best_features = (
        chromosome_to_features(
            global_best_chromosome,
            all_features,
        )
    )

    print(
        "\n===== 最良個体の再評価 ====="
    )
    print(
        f"最良 AUC: "
        f"{global_best_fitness:.6f}"
    )
    print(
        f"特徴量数: "
        f"{len(best_features)}"
    )
    print(
        "特徴量: "
        + ", ".join(best_features)
    )

    best_config = copy.deepcopy(
        config
    )
    best_config["features"] = (
        best_features.copy()
    )

    X_best = X_all[
        best_features
    ].copy()

    best_evaluation = evaluate_lightgbm(
        X_original=X_best,
        y=y,
        config=best_config,
    )

    best_result_path = (
        output_dir
        / "best_result.xlsx"
    )

    save_result(
        output_path=best_result_path,
        config=best_config,
        all_data=best_evaluation[
            "all_data"
        ],
        task_type=best_evaluation[
            "task_type"
        ],
        result_sheets=best_evaluation[
            "result_sheets"
        ],
        feature_importance=best_evaluation[
            "feature_importance"
        ],
    )

    summary_path = save_ga_summary(
        output_dir=output_dir,
        generation_history=(
            generation_history
        ),
        individual_history=(
            individual_history
        ),
        best_features=best_features,
        all_features=all_features,
        best_chromosome=(
            global_best_chromosome
        ),
        fitness_cache=fitness_cache,
    )

    print(
        f"\n最良個体の詳細: "
        f"{best_result_path}"
    )
    print(
        f"GA集計結果: "
        f"{summary_path}"
    )

    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config",
        help="設定JSONのパス",
    )

    parser.add_argument(
        "--data",
        default=None,
        help="元データのExcelファイル",
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=20,
        help="1世代の個体数",
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="世代数",
    )

    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.8,
        help="交叉率",
    )

    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=None,
        help=(
            "各遺伝子の突然変異率。"
            "省略時は 1 / 特徴量数"
        ),
    )

    parser.add_argument(
        "--elite-size",
        type=int,
        default=2,
        help="次世代へそのまま残す上位個体数",
    )

    parser.add_argument(
        "--tournament-size",
        type=int,
        default=3,
        help="トーナメント選択の個体数",
    )

    parser.add_argument(
        "--min-features",
        type=int,
        default=1,
        help="選択する最小特徴量数",
    )

    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help=(
            "選択する最大特徴量数。"
            "省略時は全特徴量数"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード",
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
        mutation_rate=(
            args.mutation_rate
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
        random_seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
