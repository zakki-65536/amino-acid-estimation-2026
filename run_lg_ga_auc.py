from __future__ import annotations

import argparse
import copy
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from filter import filter_data
from run_lightgbm import (
    DEFAULT_THRESHOLD,
    evaluate_lightgbm,
    evaluate_lightgbm_holdout,
    save_result,
)


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"

# 外側CV: 5-foldを10回反復 = 50 outer splits
OUTER_N_SPLITS = 5
OUTER_N_REPEATS = 10
OUTER_RANDOM_STATE = 42


def current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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


def get_outer_repeat_and_fold(
    split_number: int,
) -> tuple[int, int]:
    repeat_number = (
        (split_number - 1) // OUTER_N_SPLITS
    ) + 1

    fold_number = (
        (split_number - 1) % OUTER_N_SPLITS
    ) + 1

    return repeat_number, fold_number


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
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    fitness_cache: dict[tuple[int, ...], float],
    skip_threshold_cv: bool,
) -> float:
    """
    outer TRAINだけを使って候補特徴量集合をCV評価し、
    roc_auc の平均値をGAのFitnessとして返す。

    outer TESTはここでは一切使用しない。
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

    X_current = X_train[
        selected_features
    ].copy()

    try:
        evaluation = evaluate_lightgbm(
            X_original=X_current,
            y=y_train,
            config=current_config,
            skip_threshold_cv=skip_threshold_cv,
            default_threshold=DEFAULT_THRESHOLD,
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


def run_ga_on_outer_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
    all_features: list[str],
    output_dir: Path,
    population_size: int,
    generations: int,
    early_stopping_rounds: int,
    crossover_rate: float,
    mutation_rate: float,
    elite_size: int,
    tournament_size: int,
    min_features: int,
    max_features: int,
    random_seed: int,
    skip_threshold_cv: bool,
) -> tuple[
    list[str],
    float,
    np.ndarray,
    Path,
    Path,
]:
    """
    1つのouter TRAIN内だけでGAを完結させる。
    outer TESTは受け取らないため、特徴量決定に利用できない。
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    rng = np.random.default_rng(random_seed)

    feature_count = len(all_features)

    population = initialize_population(
        population_size=population_size,
        feature_count=feature_count,
        min_features=min_features,
        max_features=max_features,
        rng=rng,
    )

    fitness_cache: dict[
        tuple[int, ...],
        float,
    ] = {}

    generation_history: list[dict] = []
    individual_history: list[dict] = []

    global_best_chromosome: np.ndarray | None = None
    global_best_fitness = float("-inf")
    no_improvement_generations = 0

    for generation in range(
        1,
        generations + 1,
    ):
        print(
            f"\n  ===== Generation "
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
                X_train=X_train,
                y_train=y_train,
                config=config,
                fitness_cache=fitness_cache,
                skip_threshold_cv=skip_threshold_cv,
            )

            fitnesses.append(fitness)

            selected_features = chromosome_to_features(
                chromosome,
                all_features,
            )

            individual_history.append({
                "generation": generation,
                "individual": individual_index,
                "fitness_auc": fitness,
                "feature_count": len(selected_features),
                "features": ",".join(selected_features),
                "chromosome": "".join(
                    str(int(gene))
                    for gene in chromosome
                ),
            })

            print(
                f"  Individual {individual_index:02d}: "
                f"AUC={fitness:.6f}, "
                f"features={len(selected_features)}"
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

        best_index = int(np.argmax(fitnesses))
        generation_best_fitness = float(
            fitnesses[best_index]
        )
        generation_best_chromosome = (
            population[best_index].copy()
        )
        generation_best_features = chromosome_to_features(
            generation_best_chromosome,
            all_features,
        )

        generation_mean = float(
            np.mean(finite_fitnesses)
        )
        generation_std = float(
            np.std(finite_fitnesses, ddof=0)
        )

        improved = (
            generation_best_fitness > global_best_fitness
        )

        if improved:
            global_best_fitness = generation_best_fitness
            global_best_chromosome = (
                generation_best_chromosome.copy()
            )
            no_improvement_generations = 0
        else:
            no_improvement_generations += 1

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
            "global_best_auc": global_best_fitness,
            "improved": improved,
            "no_improvement_generations": (
                no_improvement_generations
            ),
        })

        print(
            "  世代最良 AUC: "
            f"{generation_best_fitness:.6f}"
        )
        print(
            "  世代平均 AUC: "
            f"{generation_mean:.6f} ± {generation_std:.6f}"
        )
        print(
            "  全世代最良 AUC: "
            f"{global_best_fitness:.6f}"
        )
        print(
            "  改善なし世代数: "
            f"{no_improvement_generations}/"
            f"{early_stopping_rounds}"
        )

        if (
            no_improvement_generations
            >= early_stopping_rounds
        ):
            print(
                "  早期終了: "
                f"{early_stopping_rounds}世代連続で "
                "最良AUCが改善しなかったため、"
                f"Generation {generation} で終了します。"
            )
            break

        if generation == generations:
            print(
                "  最大世代数に到達したため終了します。"
            )
            break

        sorted_indexes = sorted(
            range(len(population)),
            key=lambda i: fitnesses[i],
            reverse=True,
        )

        next_population: list[np.ndarray] = [
            population[i].copy()
            for i in sorted_indexes[:elite_size]
        ]

        while len(next_population) < population_size:
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

            child1, child2 = uniform_crossover(
                parent1=parent1,
                parent2=parent2,
                crossover_rate=crossover_rate,
                rng=rng,
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

            next_population.append(child1)

            if len(next_population) < population_size:
                next_population.append(child2)

        population = next_population

    if global_best_chromosome is None:
        raise RuntimeError(
            "最良個体を取得できませんでした。"
        )

    best_features = chromosome_to_features(
        global_best_chromosome,
        all_features,
    )

    best_config = copy.deepcopy(config)
    best_config["features"] = best_features.copy()

    # outer TRAIN内での最良特徴量のCV詳細を保存する。
    X_best_train = X_train[
        best_features
    ].copy()

    best_cv_result_path = (
        output_dir / "best_cv_result.xlsx"
    )

    best_cv_evaluation = evaluate_lightgbm(
        X_original=X_best_train,
        y=y_train,
        config=best_config,
        skip_threshold_cv=skip_threshold_cv,
        default_threshold=DEFAULT_THRESHOLD,
    )

    save_result(
        output_path=best_cv_result_path,
        config=best_config,
        all_data=best_cv_evaluation["all_data"],
        task_type=best_cv_evaluation["task_type"],
        result_sheets=best_cv_evaluation["result_sheets"],
        feature_importance=best_cv_evaluation[
            "feature_importance"
        ],
    )

    ga_summary_path = save_ga_summary(
        output_dir=output_dir,
        generation_history=generation_history,
        individual_history=individual_history,
        best_features=best_features,
        all_features=all_features,
        best_chromosome=global_best_chromosome,
        fitness_cache=fitness_cache,
    )

    return (
        best_features,
        global_best_fitness,
        global_best_chromosome,
        best_cv_result_path,
        ga_summary_path,
    )


def evaluate_outer_test_once(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_features: list[str],
    test_indexes: np.ndarray,
) -> tuple[
    float,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    GA終了後、固定した最良特徴量集合でouter TESTを1回だけ評価する。
    LightGBMの学習・予測は run_lightgbm.py 側で実行する。
    """
    evaluation = evaluate_lightgbm_holdout(
        X_train_original=X_train[best_features].copy(),
        y_train=y_train,
        X_test_original=X_test[best_features].copy(),
        y_test=y_test,
        default_threshold=DEFAULT_THRESHOLD,
    )

    predictions = evaluation["predictions"].copy()
    predictions.insert(
        0,
        "filtered_row",
        test_indexes + 1,
    )

    return (
        float(evaluation["roc_auc"]),
        predictions,
        evaluation["feature_importance"].copy(),
    )


def save_outer_fold_result(
    output_dir: Path,
    repeat_number: int,
    fold_number: int,
    train_indexes: np.ndarray,
    test_indexes: np.ndarray,
    ga_cv_best_auc: float,
    outer_test_auc: float,
    best_features: list[str],
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> Path:
    output_path = (
        output_dir / "outer_test_result.xlsx"
    )

    summary = pd.DataFrame([{
        "outer_repeat": repeat_number,
        "outer_fold": fold_number,
        "train_count": len(train_indexes),
        "test_count": len(test_indexes),
        "ga_cv_best_auc": ga_cv_best_auc,
        "outer_test_auc": outer_test_auc,
        "feature_count": len(best_features),
    }])

    best_features_df = pd.DataFrame({
        "feature": best_features,
    })

    split_data = pd.concat([
        pd.DataFrame({
            "filtered_row": train_indexes + 1,
            "split": "train",
        }),
        pd.DataFrame({
            "filtered_row": test_indexes + 1,
            "split": "test",
        }),
    ], ignore_index=True).sort_values(
        "filtered_row"
    ).reset_index(drop=True)

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


def create_feature_selection_frequency(
    feature_sets: pd.DataFrame,
    all_features: list[str],
    total_outer_splits: int,
) -> pd.DataFrame:
    counts = (
        feature_sets["feature"]
        .value_counts()
        .reindex(all_features, fill_value=0)
    )

    result = pd.DataFrame({
        "feature": all_features,
        "selected_count": [
            int(counts[feature])
            for feature in all_features
        ],
    })

    result["selection_rate"] = (
        result["selected_count"]
        / total_outer_splits
    )

    return result.sort_values(
        ["selected_count", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)


def run_genetic_algorithm_repeated_cv(
    config_path: str | Path,
    data_path: str | Path | None = None,
    population_size: int = 20,
    generations: int = 100,
    early_stopping_rounds: int = 10,
    crossover_rate: float = 0.8,
    mutation_rate: float | None = None,
    elite_size: int = 2,
    tournament_size: int = 3,
    min_features: int = 1,
    max_features: int | None = None,
    random_seed: int = 42,
    outer_random_state: int = OUTER_RANDOM_STATE,
    output_path: str | Path | None = None,
    skip_threshold_cv: bool = False,
) -> Path:
    if data_path is None:
        X_all, y_all, config = filter_data(
            config_path
        )
    else:
        X_all, y_all, config = filter_data(
            config_path,
            data_path,
        )

    if y_all.nunique() != 2:
        raise ValueError(
            "このGAプログラムのFitness=roc_aucは二値分類を前提としています。"
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

    if early_stopping_rounds < 1:
        raise ValueError(
            "early_stopping_rounds は1以上にしてください。"
        )

    if mutation_rate is None:
        mutation_rate = 1.0 / feature_count

    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError(
            "mutation_rate は0～1にしてください。"
        )

    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError(
            "crossover_rate は0～1にしてください。"
        )

    experiment_name = config["experiment_name"]

    if output_path is None:
        output_dir = (
            RESULT_DIR
            / f"GA_RepeatedCV_{experiment_name}"
        )
    else:
        output_dir = Path(output_path)

        if not output_dir.is_absolute():
            output_dir = BASE_DIR / output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outer_cv = RepeatedStratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        n_repeats=OUTER_N_REPEATS,
        random_state=outer_random_state,
    )

    total_outer_splits = (
        OUTER_N_SPLITS * OUTER_N_REPEATS
    )

    print(f"実験名: {experiment_name}")
    print(f"被験者数: {len(X_all)}")
    print(f"全特徴量数: {feature_count}")
    print(
        "外側CV: "
        f"{OUTER_N_SPLITS}-fold × "
        f"{OUTER_N_REPEATS}反復 "
        f"= {total_outer_splits}回"
    )
    print(
        "各outer splitで TRAIN約80%だけを使ってGAを実行し、"
        "TEST約20%は特徴量決定後に1回だけ評価します。"
    )
    print(f"個体数: {population_size}")
    print(f"最大世代数: {generations}")
    print(
        "早期終了条件: "
        f"{early_stopping_rounds}世代連続で最良AUCの改善なし"
    )
    print(
        "GA内部の閾値決定CV: "
        + (
            f"省略（固定閾値={DEFAULT_THRESHOLD}、AUCはpredict_proba）"
            if skip_threshold_cv
            else "実行（従来どおり）"
        )
    )
    print(f"交叉率: {crossover_rate}")
    print(f"突然変異率: {mutation_rate}")
    print(f"エリート数: {elite_size}")
    print(
        "特徴量数制約: "
        f"{min_features}～{max_features}"
    )

    outer_result_rows: list[dict] = []
    feature_set_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for split_number, (
        train_indexes,
        test_indexes,
    ) in enumerate(
        outer_cv.split(X_all, y_all),
        start=1,
    ):
        repeat_number, fold_number = (
            get_outer_repeat_and_fold(split_number)
        )

        print(
            "\n"
            + "=" * 70
        )
        print(
            f"[{current_timestamp()}] "
            f"OUTER Repeat {repeat_number}/{OUTER_N_REPEATS}, "
            f"Fold {fold_number}/{OUTER_N_SPLITS} "
            f"({split_number}/{total_outer_splits})"
        )
        print("=" * 70)

        X_train = (
            X_all.iloc[train_indexes]
            .reset_index(drop=True)
        )
        X_test = (
            X_all.iloc[test_indexes]
            .reset_index(drop=True)
        )
        y_train = (
            y_all.iloc[train_indexes]
            .reset_index(drop=True)
        )
        y_test = (
            y_all.iloc[test_indexes]
            .reset_index(drop=True)
        )

        print(
            f"outer TRAIN: {len(X_train)}件 / "
            f"outer TEST: {len(X_test)}件"
        )

        fold_output_dir = (
            output_dir
            / f"repeat_{repeat_number:02d}"
            / f"fold_{fold_number:02d}"
        )
        fold_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 各outer splitでGAを独立実行する。
        # splitごとにseedをずらして、GA探索も独立させる。
        fold_ga_seed = (
            random_seed + split_number - 1
        )

        (
            best_features,
            ga_cv_best_auc,
            _best_chromosome,
            _best_cv_result_path,
            _ga_summary_path,
        ) = run_ga_on_outer_train(
            X_train=X_train,
            y_train=y_train,
            config=config,
            all_features=all_features,
            output_dir=fold_output_dir,
            population_size=population_size,
            generations=generations,
            early_stopping_rounds=early_stopping_rounds,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
            tournament_size=tournament_size,
            min_features=min_features,
            max_features=max_features,
            random_seed=fold_ga_seed,
            skip_threshold_cv=skip_threshold_cv,
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
            best_features=best_features,
            test_indexes=test_indexes,
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
        prediction_frames.append(predictions)

        save_outer_fold_result(
            output_dir=fold_output_dir,
            repeat_number=repeat_number,
            fold_number=fold_number,
            train_indexes=train_indexes,
            test_indexes=test_indexes,
            ga_cv_best_auc=ga_cv_best_auc,
            outer_test_auc=outer_test_auc,
            best_features=best_features,
            predictions=predictions,
            feature_importance=feature_importance,
        )

        outer_result_rows.append({
            "outer_repeat": repeat_number,
            "outer_fold": fold_number,
            "outer_split": split_number,
            "ga_seed": fold_ga_seed,
            "skip_threshold_cv": skip_threshold_cv,
            "train_count": len(train_indexes),
            "test_count": len(test_indexes),
            "ga_cv_best_auc": ga_cv_best_auc,
            "outer_test_auc": outer_test_auc,
            "feature_count": len(best_features),
            "features": ",".join(best_features),
        })

        feature_set_frames.append(
            pd.DataFrame({
                "outer_repeat": repeat_number,
                "outer_fold": fold_number,
                "outer_split": split_number,
                "feature": best_features,
            })
        )

        print("\n  ----- outer TEST評価 -----")
        print(
            f"  GA CV 最良AUC: {ga_cv_best_auc:.6f}"
        )
        print(
            f"  outer TEST AUC: {outer_test_auc:.6f}"
        )
        print(
            f"  選択特徴量数: {len(best_features)}"
        )

    outer_results = pd.DataFrame(
        outer_result_rows
    )
    feature_sets = pd.concat(
        feature_set_frames,
        ignore_index=True,
    )
    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    feature_frequency = (
        create_feature_selection_frequency(
            feature_sets=feature_sets,
            all_features=all_features,
            total_outer_splits=total_outer_splits,
        )
    )

    # 50個のfold AUCの集計
    fold_auc_summary = pd.DataFrame([{
        "outer_fold_auc_mean": (
            outer_results["outer_test_auc"].mean()
        ),
        "outer_fold_auc_std": (
            outer_results["outer_test_auc"].std()
        ),
        "outer_fold_auc_min": (
            outer_results["outer_test_auc"].min()
        ),
        "outer_fold_auc_max": (
            outer_results["outer_test_auc"].max()
        ),
        "outer_fold_auc_median": (
            outer_results["outer_test_auc"].median()
        ),
        "ga_cv_best_auc_mean": (
            outer_results["ga_cv_best_auc"].mean()
        ),
        "selected_feature_count_mean": (
            outer_results["feature_count"].mean()
        ),
        "selected_feature_count_std": (
            outer_results["feature_count"].std()
        ),
        "outer_splits": total_outer_splits,
        "outer_n_splits": OUTER_N_SPLITS,
        "outer_n_repeats": OUTER_N_REPEATS,
        "outer_random_state": outer_random_state,
        "skip_threshold_cv": skip_threshold_cv,
    }])

    # 各repeatについて5fold分のOOF予測をまとめたAUCを算出する。
    repeat_rows = []

    for repeat_number in range(
        1,
        OUTER_N_REPEATS + 1,
    ):
        repeat_predictions = all_predictions[
            all_predictions["outer_repeat"]
            == repeat_number
        ]

        repeat_auc = float(
            roc_auc_score(
                repeat_predictions["actual"],
                repeat_predictions[
                    "probability_positive"
                ],
            )
        )

        repeat_rows.append({
            "outer_repeat": repeat_number,
            "oof_auc": repeat_auc,
        })

    repeat_auc_df = pd.DataFrame(
        repeat_rows
    )

    repeat_auc_summary = pd.DataFrame([{
        "repeat_oof_auc_mean": (
            repeat_auc_df["oof_auc"].mean()
        ),
        "repeat_oof_auc_std": (
            repeat_auc_df["oof_auc"].std()
        ),
        "repeat_oof_auc_min": (
            repeat_auc_df["oof_auc"].min()
        ),
        "repeat_oof_auc_max": (
            repeat_auc_df["oof_auc"].max()
        ),
        "repeat_count": OUTER_N_REPEATS,
    }])

    summary_output_path = (
        output_dir / "outer_cv_summary.xlsx"
    )

    with pd.ExcelWriter(
        summary_output_path,
        engine="openpyxl",
    ) as writer:
        fold_auc_summary.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )
        repeat_auc_summary.to_excel(
            writer,
            sheet_name="repeat_summary",
            index=False,
        )
        repeat_auc_df.to_excel(
            writer,
            sheet_name="repeat_auc",
            index=False,
        )
        outer_results.to_excel(
            writer,
            sheet_name="outer_fold_results",
            index=False,
        )
        feature_sets.to_excel(
            writer,
            sheet_name="feature_sets",
            index=False,
        )
        feature_frequency.to_excel(
            writer,
            sheet_name="feature_frequency",
            index=False,
        )
        all_predictions.to_excel(
            writer,
            sheet_name="test_predictions",
            index=False,
        )

    print("\n" + "=" * 70)
    print("===== 外側5-fold × 10反復 最終結果 =====")
    print("=" * 70)
    print(
        "50 foldのouter TEST AUC: "
        f"{fold_auc_summary.iloc[0]['outer_fold_auc_mean']:.6f} "
        "± "
        f"{fold_auc_summary.iloc[0]['outer_fold_auc_std']:.6f}"
    )
    print(
        "10 repeatのOOF AUC: "
        f"{repeat_auc_summary.iloc[0]['repeat_oof_auc_mean']:.6f} "
        "± "
        f"{repeat_auc_summary.iloc[0]['repeat_oof_auc_std']:.6f}"
    )
    print(
        "平均選択特徴量数: "
        f"{fold_auc_summary.iloc[0]['selected_feature_count_mean']:.2f}"
    )
    print(
        f"\n集計結果保存先: {summary_output_path}"
    )

    return summary_output_path


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
        default=100,
        help="最大世代数（デフォルト100）",
    )

    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=10,
        help=(
            "最良AUCが改善しない状態が何世代続いたら"
            "早期終了するか（デフォルト10）"
        ),
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
        help=(
            "GAの基準乱数シード。"
            "outer splitごとにこの値から1ずつ増加"
        ),
    )

    parser.add_argument(
        "--outer-seed",
        type=int,
        default=OUTER_RANDOM_STATE,
        help="外側RepeatedStratifiedKFoldの乱数シード",
    )

    parser.add_argument(
        "--skip-threshold-cv",
        action="store_true",
        help=(
            "GA内部のAUC評価で閾値決定用CVを省略する。"
            "run_lightgbm.py 側で固定閾値0.5を使用し、"
            "AUCはpredict_probaから計算する。"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="結果保存ディレクトリ",
    )

    args = parser.parse_args()

    run_genetic_algorithm_repeated_cv(
        config_path=args.config,
        data_path=args.data,
        population_size=args.population_size,
        generations=args.generations,
        early_stopping_rounds=args.early_stopping_rounds,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elite_size=args.elite_size,
        tournament_size=args.tournament_size,
        min_features=args.min_features,
        max_features=args.max_features,
        random_seed=args.seed,
        outer_random_state=args.outer_seed,
        output_path=args.output,
        skip_threshold_cv=args.skip_threshold_cv,
    )


if __name__ == "__main__":
    main()
