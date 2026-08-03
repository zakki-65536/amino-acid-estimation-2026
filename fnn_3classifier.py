# 分類（0/1/2 予測）版
# インストールは "pip install pandas tensorflow scikit-learn openpyxl numpy matplotlib"

import os
import warnings
from datetime import datetime

# TensorFlow のログを抑制（評価結果だけを標準出力に出すため）
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")  # 画面表示なしで画像保存するため
import matplotlib.pyplot as plt


# ====== 設定 ======
excel_path = 'data/data_13項目_空腹時_male_3段階.xlsx'  # 入力Excel

N_SPLITS = 5          # k分割交差検証の k
N_REPEATS = 10        # k分割交差検証を実行する回数
RANDOM_STATE = 42

EPOCHS = 100          # 最大エポック数。EarlyStoppingにより途中で終了する場合あり
BATCH_SIZE = 16

VALIDATION_SPLIT = 0.2

EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.0001

LABELS = [0, 1, 2]


# ====== 学習進行確認用設定 ======
SHOW_TRAINING_PROGRESS = True   # Trueでepochごとの学習ログを表示
PLOT_HISTORY = True             # Trueでloss/accuracyグラフを保存
PLOT_ONLY_FIRST_FOLD = True     # Trueなら最初のrepeat=1, fold=1だけ保存
HISTORY_DIR = "result"   # グラフ保存先フォルダ


# 出力形式（ヘッダーは出力しない）:
# timestamp,k,repeat,cm00,cm01,cm02,cm10,cm11,cm12,cm20,cm21,cm22,accuracy,precision,recall,epochs_run
# precision / recall は3クラスの macro 平均
# epochs_run は、そのfoldの学習が終了した時点の実行エポック数


def build_model(input_dim):
    """3クラス分類モデルを新規作成する。"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def save_training_plots(history, repeat_no, k_no):
    """loss / accuracy の学習曲線をPNG保存する。"""

    os.makedirs(HISTORY_DIR, exist_ok=True)

    # ====== lossグラフ ======
    plt.figure()
    plt.plot(history.history["loss"], label="train loss")

    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val loss")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss repeat={repeat_no}, fold={k_no}")
    plt.legend()
    plt.grid()

    loss_plot_path = os.path.join(
        HISTORY_DIR,
        f"loss_repeat{repeat_no}_fold{k_no}.png"
    )

    plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ====== accuracyグラフ ======
    plt.figure()
    plt.plot(history.history["accuracy"], label="train accuracy")

    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="val accuracy")

    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"Accuracy repeat={repeat_no}, fold={k_no}")
    plt.legend()
    plt.grid()

    acc_plot_path = os.path.join(
        HISTORY_DIR,
        f"accuracy_repeat{repeat_no}_fold{k_no}.png"
    )

    plt.savefig(acc_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return loss_plot_path, acc_plot_path


def print_training_summary(history, repeat_no, k_no, epochs_run):
    """学習履歴の概要を標準出力に表示する。"""

    print()
    print("========== Training summary ==========")
    print(f"repeat={repeat_no}, fold={k_no}")
    print(f"epochs_run={epochs_run}")
    print(f"history keys={list(history.history.keys())}")

    print(f"final loss={history.history['loss'][-1]:.6f}")
    print(f"final accuracy={history.history['accuracy'][-1]:.6f}")

    if "val_loss" in history.history:
        print(f"final val_loss={history.history['val_loss'][-1]:.6f}")
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1
        best_val_loss = float(np.min(history.history["val_loss"]))
        print(f"best val_loss epoch={best_epoch}")
        print(f"best val_loss={best_val_loss:.6f}")

    if "val_accuracy" in history.history:
        print(f"final val_accuracy={history.history['val_accuracy'][-1]:.6f}")
        best_val_acc_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
        best_val_acc = float(np.max(history.history["val_accuracy"]))
        print(f"best val_accuracy epoch={best_val_acc_epoch}")
        print(f"best val_accuracy={best_val_acc:.6f}")

    print("======================================")
    print()


# ====== 出力フォルダ作成 ======
if PLOT_HISTORY:
    os.makedirs(HISTORY_DIR, exist_ok=True)


# ====== データ読み込み ======
data = pd.read_excel(excel_path)


# ====== 特徴量と目的変数の設定 ======
# 特徴量: 1〜13列目 → 0:13（0〜12の13列）
X = data.iloc[:, 0:13].copy()

# 14列目（index=13）は学習・評価に使わないためスキップ

# 目的変数: 15列目 → index=14（値は0/1/2）
y = data.iloc[:, 14].astype(int)


# ====== 前処理 ======
# 非数値 → NaN、inf処理、欠損は列中央値で補完
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))


# ====== k分割交差検証を10回実行 ======
# 各repeatでshuffleの乱数を変え、5分割を作り直す。
for repeat_no in range(1, N_REPEATS + 1):

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE + repeat_no
    )

    for k_no, (train_index, test_index) in enumerate(skf.split(X, y), start=1):

        print()
        print(f"===== repeat {repeat_no}/{N_REPEATS}, fold {k_no}/{N_SPLITS} =====")

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        # 特徴量の標準化は、各foldの学習データだけでfitする。
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        # foldごとにモデルを作り直す。
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(RANDOM_STATE + repeat_no * 100 + k_no)
        model = build_model(X_train_scaled.shape[1])

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        )

        class_weight = {
            0: 1.32,
            1: 0.67,
            2: 1.32
        }

        history = model.fit(
            X_train_scaled,
            y_train.values,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=[early_stopping],
            verbose=2 if SHOW_TRAINING_PROGRESS else 0,
            class_weight=class_weight
        )

        epochs_run = len(history.epoch)

        # ====== 学習履歴の確認 ======
        if SHOW_TRAINING_PROGRESS:
            print_training_summary(
                history=history,
                repeat_no=repeat_no,
                k_no=k_no,
                epochs_run=epochs_run
            )

        # ====== 学習曲線の保存 ======
        if PLOT_HISTORY:
            should_plot = True

            if PLOT_ONLY_FIRST_FOLD:
                should_plot = (repeat_no == 1 and k_no == 1)

            if should_plot:
                loss_plot_path, acc_plot_path = save_training_plots(
                    history=history,
                    repeat_no=repeat_no,
                    k_no=k_no
                )

                print(f"saved loss plot: {loss_plot_path}")
                print(f"saved accuracy plot: {acc_plot_path}")

        # model.predict() の retracing 警告を減らすため、直接呼び出しで予測する。
        proba = model(X_test_scaled, training=False).numpy()
        y_pred = np.argmax(proba, axis=1)

        # 混同行列: 行=正解, 列=予測。labels固定により必ず3x3=9値にする。
        cm = confusion_matrix(y_test.values, y_pred, labels=LABELS)
        cm_values = cm.ravel()

        acc = accuracy_score(y_test.values, y_pred)

        precision = precision_score(
            y_test.values,
            y_pred,
            labels=LABELS,
            average='macro',
            zero_division=0
        )

        recall = recall_score(
            y_test.values,
            y_pred,
            labels=LABELS,
            average='macro',
            zero_division=0
        )

        timestamp = datetime.now().isoformat(timespec='seconds')

        output_values = [
            timestamp,
            k_no,
            repeat_no,
            *cm_values.tolist(),
            f"{acc:.6f}",
            f"{precision:.6f}",
            f"{recall:.6f}",
            epochs_run,
        ]

        print(",".join(map(str, output_values)))