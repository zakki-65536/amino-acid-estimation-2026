"""filter.py 対応 LightGBM 48時間以内を目標とする拡張 Grid Search。

探索候補: 294,912通り × StratifiedKFold(5分割,1回)
データ: 学習用80%で最適化、独立TEST20%は最後の一度だけ評価。
実行順: 固定シードでランダム化。時間切れでも特定のパラメータに偏りにくい。
停止: --max-hours の既定47時間（結果出力・最終学習用の余裕1時間）。
保存: 評価ごとCSVへ追記。--resumeで未評価候補から再開。

必要: pip install lightgbm scikit-learn pandas numpy openpyxl joblib
実行例:
  python grid_search_lightgbm_48h.py config.json --threads 4
  python grid_search_lightgbm_48h.py config.json --threads 4 --max-candidates 30
  python grid_search_lightgbm_48h.py config.json --threads 4 --resume

注意: --max-candidatesによる中断は最終TEST評価をしません。
      時間切れまたは全探索終了で最良モデルを選んでTEST評価し、探索を確定します。
      時間上限はPythonレベルのソフトリミットで、学習中の処理を強制終了しません。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from filter import (
    DEFAULT_DATA_PATH, apply_filters, load_config, load_data,
    discretize_by_ratios, discretize_by_cutoffs, normalize_item_id, resolve_path,
)

# 6組から12組へ拡張。旧6組をすべて含む。
LEARNING_PAIRS = [
    (100, 0.15), (150, 0.15), (200, 0.10), (250, 0.10),
    (400, 0.07), (400, 0.10), (600, 0.05), (600, 0.03),
    (800, 0.03), (800, 0.05), (1000, 0.02), (1200, 0.02),
]
# 5種類の深さ、合計16組。num_leavesは2**max_depth以下。
DEPTH_LEAVES = {
    3: [3, 7],
    4: [7, 15],
    6: [7, 15, 31, 63],
    8: [7, 15, 31, 63, 127],
    10: [31, 63, 127],
}
MIN_CHILD_SAMPLES = [5, 10, 20, 30, 60, 100]
SUBSAMPLE = [0.7, 0.8, 0.9, 1.0]
COLSAMPLE_BYTREE = [0.7, 0.8, 0.9, 1.0]
REG_ALPHA = [0.0, 0.1, 1.0, 10.0]
REG_LAMBDA = [0.0, 0.1, 1.0, 10.0]

GRID_AXES = [
    LEARNING_PAIRS,
    [(d, leaves) for d, choices in DEPTH_LEAVES.items() for leaves in choices],
    MIN_CHILD_SAMPLES, SUBSAMPLE, COLSAMPLE_BYTREE, REG_ALPHA, REG_LAMBDA,
]
TOTAL_CANDIDATES = math.prod(map(len, GRID_AXES))
assert TOTAL_CANDIDATES == 294_912
GRID_COLUMNS = [
    "n_estimators", "learning_rate", "max_depth", "num_leaves",
    "min_child_samples", "subsample", "colsample_bytree", "reg_alpha",
    "reg_lambda",
]
RESULT_COLUMNS = [
    "candidate_id", *GRID_COLUMNS, "cv_mean_auc", "cv_std_auc",
    "fold_1_auc", "fold_2_auc", "fold_3_auc", "fold_4_auc", "fold_5_auc",
    "elapsed_sec",
]


def candidate_by_id(candidate_id: int) -> dict:
    """固定グリッドの候補番号から定数時間でパラメータを生成（巨大配列不要）。"""
    if not 1 <= candidate_id <= TOTAL_CANDIDATES:
        raise ValueError(f"候補IDは1〜{TOTAL_CANDIDATES}です")
    n = candidate_id - 1
    indices = []
    for axis in reversed(GRID_AXES):
        n, rem = divmod(n, len(axis))
        indices.append(rem)
    values = [axis[j] for axis, j in zip(GRID_AXES, reversed(indices))]
    (estimators, rate), (depth, leaves), min_child, sub, col, alpha, lamb = values
    return dict(
        n_estimators=estimators, learning_rate=rate, max_depth=depth,
        num_leaves=leaves, min_child_samples=min_child, subsample=sub,
        colsample_bytree=col, reg_alpha=alpha, reg_lambda=lamb,
    )


def grid_description() -> dict:
    return {
        "learning_pairs": LEARNING_PAIRS,
        "depth_leaves": DEPTH_LEAVES,
        "min_child_samples": MIN_CHILD_SAMPLES,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "reg_alpha": REG_ALPHA,
        "reg_lambda": REG_LAMBDA,
        "candidates": TOTAL_CANDIDATES,
        "candidate_order": "numpy.default_rng(seed).permutation(candidate_ids)",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="filter.pyを用いるLightGBM 294,912通りGrid Search（既定47時間）"
    )
    parser.add_argument("config", help="filter.pyの設定JSON")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="元Excelファイル")
    parser.add_argument("--output-dir", default="result/lightgbm_grid_48h", help="保存先")
    parser.add_argument("--seed", type=int, default=42, help="分割・LightGBM・候補順の乱数シード")
    parser.add_argument("--test-size", type=float, default=0.2, help="独立TEST比率")
    parser.add_argument("--threads", type=int, default=4, help="LightGBM内部スレッド数")
    parser.add_argument("--max-hours", type=float, default=47.0,
                        help="探索時間の上限（既定47時間、最終処理用に約1時間確保）")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="今回の実行で追加評価する最大候補数（試運転用）")
    parser.add_argument("--resume", action="store_true", help="同じ設定で探索を再開")
    parser.add_argument("--progress-every", type=int, default=500, help="進捗表示間隔")
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-sizeは0より大きく1未満")
    if args.threads < 1:
        parser.error("--threadsは1以上")
    if args.max_hours <= 0 or not math.isfinite(args.max_hours):
        parser.error("--max-hoursは正の有限値")
    if args.max_candidates is not None and args.max_candidates < 1:
        parser.error("--max-candidatesは1以上")
    if args.progress_every < 1:
        parser.error("--progress-everyは1以上")
    return args


def make_data(args):
    """Excelを一度だけ読み、filter.pyの関数で抽出・分類する。"""
    config = load_config(args.config)
    filtered = apply_filters(load_data(args.data), config["filters"])
    features = [normalize_item_id(item_id) for item_id in config["features"]]
    target_config = config["target"]
    target_id = normalize_item_id(target_config["id"])
    X = filtered[features].copy().reset_index(drop=True)
    raw_y = filtered[target_id].copy().reset_index(drop=True)
    if X.empty or X.shape[1] == 0:
        raise ValueError("対象データまたは特徴量が0件です")
    if X.columns.duplicated().any():
        raise ValueError("featuresに重複する項目があります")
    if target_id in X.columns:
        raise ValueError("targetがfeaturesにも含まれています（情報漏洩）")
    ratios, cutoffs = target_config.get("ratios"), target_config.get("cutoffs")
    if (ratios is None) == (cutoffs is None):
        raise ValueError("targetにはratiosかcutoffsのどちらか一方を指定してください")
    if raw_y.isna().any():
        raise ValueError(f"目的変数に欠損値が{raw_y.isna().sum()}件あります")

    ratio_array = None
    if ratios is not None:
        ratio_array = np.asarray(ratios, dtype=float)
        if (len(ratio_array) < 2 or not np.isfinite(ratio_array).all()
                or (ratio_array <= 0).any()):
            raise ValueError("ratiosは2個以上の正の有限値が必要です")
        initial_y = discretize_by_ratios(raw_y, ratios)
    else:
        initial_y = discretize_by_cutoffs(raw_y, cutoffs)
    if initial_y.nunique() < 2 or initial_y.value_counts().min() < 2:
        raise ValueError("分類後のクラス件数が少なすぎます")

    # 元のfixed版との比較可能性を保つため初期ラベルで層化分割する。
    X_train, X_test, ytr0, yte0, raw_train, raw_test = train_test_split(
        X, initial_y, raw_y, test_size=args.test_size,
        stratify=initial_y, random_state=args.seed,
    )
    boundaries = None
    if ratios is not None:
        quantiles = np.cumsum(ratio_array)[:-1] / ratio_array.sum()
        boundaries = np.quantile(raw_train.to_numpy(), quantiles).tolist()
        if len(set(boundaries)) != len(boundaries):
            raise ValueError("ratiosの境界値が重複しました。cutoffsを検討してください")
        y_train = np.digitize(raw_train.to_numpy(), bins=boundaries)
        y_test = np.digitize(raw_test.to_numpy(), bins=boundaries)
    else:
        y_train, y_test = ytr0.to_numpy(), yte0.to_numpy()

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    try:
        y_test = encoder.transform(y_test)
    except ValueError as exc:
        raise ValueError("TESTだけに出現するクラスがあります") from exc
    n_classes = len(encoder.classes_)
    train_counts = np.bincount(y_train, minlength=n_classes)
    test_counts = np.bincount(y_test, minlength=n_classes)
    if n_classes < 2 or min(train_counts) < 5:
        raise ValueError("5分割CVには学習データの各クラスに5件以上必要です")
    if min(test_counts) == 0:
        raise ValueError("TESTにないクラスがあります")
    return (
        X_train.reset_index(drop=True), X_test.reset_index(drop=True),
        np.asarray(y_train), np.asarray(y_test), np.asarray(raw_train),
        config, [int(v) for v in encoder.classes_], boundaries,
        train_counts.tolist(), test_counts.tolist(),
    )


def make_folds(X_train, y_train, raw_train, config, n_classes, seed):
    ratios = config["target"].get("ratios")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    prepared = []
    for tr, va in splitter.split(X_train, y_train):
        if ratios is None:
            ytr, yva = y_train[tr], y_train[va]
        else:
            r = np.asarray(ratios, dtype=float)
            q = np.cumsum(r)[:-1] / r.sum()
            bounds = np.quantile(raw_train[tr], q)
            if len(set(bounds.tolist())) != len(bounds):
                raise ValueError("CV中のratios閾値が重複しました")
            ytr = np.digitize(raw_train[tr], bins=bounds)
            yva = np.digitize(raw_train[va], bins=bounds)
        if len(np.unique(ytr)) != n_classes or len(np.unique(yva)) != n_classes:
            raise ValueError("CV中にクラス欠落があります")
        prepared.append((X_train.iloc[tr], X_train.iloc[va], ytr, yva))
    return prepared


def auc(y_true, prob, n_classes):
    if n_classes == 2:
        return float(roc_auc_score(y_true, prob[:, 1]))
    return float(roc_auc_score(
        y_true, prob, multi_class="ovr", average="macro",
        labels=list(range(n_classes)),
    ))


def fixed_model_params(seed, threads, n_classes):
    return {
        "boosting_type": "gbdt", "objective": "binary" if n_classes == 2 else "multiclass",
        "random_state": seed, "n_jobs": threads, "verbosity": -1,
        "subsample_freq": 1,
    }


class DeadlineReached(Exception):
    pass


def evaluate_candidate(params, fixed_params, folds, n_classes, deadline):
    scores = []
    for xtr, xva, ytr, yva in folds:
        if time.monotonic() >= deadline:
            raise DeadlineReached
        model = lgb.LGBMClassifier(**fixed_params, **params)
        model.fit(xtr, ytr)
        scores.append(auc(yva, model.predict_proba(xva), n_classes))
    return scores


def evaluate_test(model, X_test, y_test, labels):
    n_classes = len(labels)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)
    averaging = "binary" if n_classes == 2 else "macro"
    return {
        "auc": auc(y_test, prob, n_classes),
        "f1": float(f1_score(y_test, pred, average=averaging, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, average=averaging, zero_division=0)),
        "recall": float(recall_score(y_test, pred, average=averaging, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_test, pred, labels=list(range(n_classes)),
        ).tolist(),
        "classification_report": classification_report(
            y_test, pred, labels=list(range(n_classes)),
            target_names=[str(v) for v in labels],
            output_dict=True, zero_division=0,
        ),
    }


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, content: dict):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_completed(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError("--resume指定時にgrid_results.csvがありません")
    ids = set()
    best_id, best_auc, best_row = None, -math.inf, None
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != RESULT_COLUMNS:
            raise ValueError("CSVの列名が違います。出力先を確認してください")
        for row in reader:
            i = int(row["candidate_id"])
            if i in ids:
                raise ValueError(f"重複する候補ID: {i}")
            expected = candidate_by_id(i)
            for key in GRID_COLUMNS:
                if not math.isclose(
                    float(row[key]), float(expected[key]), rel_tol=1e-12, abs_tol=1e-12,
                ):
                    raise ValueError(f"保存済み候補#{i}の{key}が一致しません")
            score = float(row["cv_mean_auc"])
            if not math.isfinite(score):
                raise ValueError("保存済みのAUCが有限値ではありません")
            ids.add(i)
            if score > best_auc or (score == best_auc and (best_id is None or i < best_id)):
                best_auc, best_id, best_row = score, i, row
    return ids, best_id, best_auc, best_row


def main():
    args = parse_args()
    start = time.monotonic()
    deadline = start + args.max_hours * 3600
    config_path, data_path = resolve_path(args.config), resolve_path(args.data)
    out = resolve_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run_config.json"
    csv_path = out / "grid_results.csv"
    final_path = out / "result.json"

    signature = {
        "script": "lightgbm_grid_48h_v1",
        "config_path": str(config_path.resolve()),
        "data_path": str(data_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "data_sha256": sha256_file(data_path),
        "seed": args.seed, "test_size": args.test_size, "threads": args.threads,
        "grid_definition": grid_description(), "cv_folds": 5, "cv_repeats": 1,
    }
    # JSON保存後にtuple->list / intキー->strキーとなるため、再開時も同じ形式で比較する。
    signature = json.loads(json.dumps(signature))
    if args.resume:
        if final_path.exists():
            raise RuntimeError("最終TEST評価済みの探索は再開できません（評価漏洩防止）")
        if not manifest_path.exists():
            raise FileNotFoundError("--resume指定時にrun_config.jsonがありません")
        with manifest_path.open(encoding="utf-8") as f:
            old_signature = json.load(f)
        if old_signature != signature:
            raise ValueError("データ、設定、スレッド数、乱数シード、探索グリッドが異なり再開できません")
        done, best_id, best_auc, best_row = load_completed(csv_path)
    else:
        if manifest_path.exists() or csv_path.exists() or final_path.exists():
            raise FileExistsError(
                f"出力先 {out} に結果があります。--resumeまたは別の--output-dirを使ってください"
            )
        done, best_id, best_auc, best_row = set(), None, -math.inf, None

    (
        X_train, X_test, y_train, y_test, raw_train, config,
        labels, boundaries, train_counts, test_counts,
    ) = make_data(args)
    folds = make_folds(X_train, y_train, raw_train, config, len(labels), args.seed)
    fixed_params = fixed_model_params(args.seed, args.threads, len(labels))
    if not args.resume:
        write_json(manifest_path, signature)

    print(f"実験: {config.get('experiment_name', '未指定')}")
    print(f"train={len(y_train)}, TEST={len(y_test)}, 特徴量数={X_train.shape[1]}")
    print(f"クラス={labels}, 学習クラス件数={train_counts}, TESTクラス件数={test_counts}")
    if boundaries is not None:
        print(f"trainから計算したratios境界: {boundaries}")
    print(f"候補数={TOTAL_CANDIDATES:,}, CV学習最大={TOTAL_CANDIDATES * 5:,}回")
    print(f"既完了={len(done):,}件 / 今回の上限={args.max_hours}時間")
    print(f"出力先: {out}", flush=True)

    # 候補を固定シードでシャッフルし、時間切れ時の偏りを小さくする。
    shuffled_ids = np.random.default_rng(args.seed).permutation(TOTAL_CANDIDATES) + 1
    new_done = 0
    timed_out = False
    limited_for_test = False
    mode = "a" if args.resume else "w"
    try:
        with csv_path.open(mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            if not args.resume:
                writer.writeheader()
                f.flush()
            for id_value in shuffled_ids:
                i = int(id_value)
                if i in done:
                    continue
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                if args.max_candidates is not None and new_done >= args.max_candidates:
                    limited_for_test = True
                    break
                params = candidate_by_id(i)
                began = time.monotonic()
                try:
                    scores = evaluate_candidate(params, fixed_params, folds, len(labels), deadline)
                except DeadlineReached:
                    timed_out = True
                    break
                mean_auc = float(np.mean(scores))
                row = {
                    "candidate_id": i, **params,
                    "cv_mean_auc": mean_auc, "cv_std_auc": float(np.std(scores)),
                    **{f"fold_{k+1}_auc": score for k, score in enumerate(scores)},
                    "elapsed_sec": time.monotonic() - began,
                }
                writer.writerow(row)
                f.flush()
                done.add(i)
                new_done += 1
                if mean_auc > best_auc or (mean_auc == best_auc and (best_id is None or i < best_id)):
                    best_auc, best_id, best_row = mean_auc, i, row
                if new_done == 1 or new_done % args.progress_every == 0 or len(done) == TOTAL_CANDIDATES:
                    elapsed = time.monotonic() - start
                    pace = elapsed / max(new_done, 1)
                    print(
                        f"完了 {len(done):,}/{TOTAL_CANDIDATES:,} | 今回 {new_done:,} | "
                        f"最高CV AUC={best_auc:.6f} (#{best_id}) | "
                        f"経過={elapsed / 3600:.2f}h | "
                        f"現ペースの残り推定={((TOTAL_CANDIDATES-len(done))*pace)/3600:.1f}h",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("\n中断しました。完了済みの候補はCSVに保存されています。--resumeで再開してください。")
        return

    if limited_for_test:
        print(f"試運転として {new_done:,}通り追加評価。最終TESTはまだ評価しません。", flush=True)
        print(f"続ける場合は同じコマンドに --resume を追加してください: {out}")
        return
    if not done:
        print("評価できた候補がありません。--resumeで再開してください。", flush=True)
        return

    # 全件完了 or 指定時間切れで探索を確定。TESTはここで一度だけ評価。
    completed_all = len(done) == TOTAL_CANDIDATES
    best_params = candidate_by_id(best_id)
    print(
        f"\n探索終了（{'全件完了' if completed_all else '時間上限'}）: "
        f"{len(done):,}/{TOTAL_CANDIDATES:,}件。最良候補 #{best_id}, CV AUC={best_auc:.6f}",
        flush=True,
    )
    model = lgb.LGBMClassifier(**fixed_params, **best_params)
    model.fit(X_train, y_train)
    test_metrics = evaluate_test(model, X_test, y_test, labels)
    selected = {**fixed_params, **best_params}
    result = {
        "experiment_name": config.get("experiment_name", ""),
        "selection_metric": "mean_5fold_cv_auc",
        "grid_total": TOTAL_CANDIDATES,
        "evaluated_candidates": len(done),
        "search_completed": completed_all,
        "stopped_by_time_limit": timed_out,
        "cv_folds": 5, "cv_repeats": 1,
        "train_size": len(y_train), "test_size_count": len(y_test),
        "test_fraction": args.test_size, "seed": args.seed,
        "feature_ids": X_train.columns.tolist(),
        "class_labels_original": labels,
        "train_class_counts": train_counts, "test_class_counts": test_counts,
        "ratio_boundaries_train_only": boundaries,
        "best_candidate_id": best_id,
        "best_cv_mean_auc": float(best_row["cv_mean_auc"]),
        "best_cv_std_auc": float(best_row["cv_std_auc"]),
        "best_cv_fold_auc": [float(best_row[f"fold_{k}_auc"]) for k in range(1, 6)],
        "best_params": selected,
        "test_metrics": test_metrics,
        "elapsed_hours": (time.monotonic() - start) / 3600,
    }
    write_json(out / "best_params.json", selected)
    joblib.dump(model, out / "best_model.joblib")
    # 大きなCSVでも最終段階でのみソートして出力する。
    df = pd.read_csv(csv_path)
    df.sort_values(["cv_mean_auc", "candidate_id"], ascending=[False, True]).to_csv(
        out / "grid_results_ranked.csv", index=False,
    )
    write_json(final_path, result)
    print(f"TEST AUC: {test_metrics['auc']:.6f}")
    print(f"TEST F1 : {test_metrics['f1']:.6f}")
    print(f"最適パラメータ: {json.dumps(selected, ensure_ascii=False)}")
    print(f"実行時間: {(time.monotonic() - start)/3600:.2f}時間", flush=True)


if __name__ == "__main__":
    main()
