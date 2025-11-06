
from __future__ import annotations
import os
from pathlib import Path
import argparse
import tensorflow as tf


os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from src.data_ucf101_tfds import make_sequence_datasets_tfds as make_sequence_datasets

ROOT = Path.home() / "bhome" / "group8"
LOG_DIR = ROOT / "logs" / "i3d_baseline"
CKPT_DIR = ROOT / "checkpoints" / "i3d_baseline"

def build_3dcnn(num_classes: int, T: int, width_mult: float = 1.0) -> tf.keras.Model:
    ch = lambda c: max(8, int(c * width_mult))
    inputs = tf.keras.Input(shape=(T, 224, 224, 3), name="clip")

    x = tf.keras.layers.Conv3D(ch(64), (3,7,7), strides=(1,2,2), padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1,3,3), strides=(1,2,2), padding="same")(x)

    x = tf.keras.layers.Conv3D(ch(128), (3,3,3), strides=(1,1,1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(2,3,3), strides=(2,2,2), padding="same")(x)  

    for _ in range(2):
        y = tf.keras.layers.Conv3D(ch(128), (3,3,3), padding="same", use_bias=False)(x)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.ReLU()(y)
        y = tf.keras.layers.Conv3D(ch(128), (3,3,3), padding="same", use_bias=False)(y)
        y = tf.keras.layers.BatchNormalization()(y)
        x = tf.keras.layers.ReLU()(x + y)

    x = tf.keras.layers.Conv3D(ch(256), (3,3,3), strides=(2,2,2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="i3d_baseline")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=32, help="sequence length T")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width_mult", type=float, default=1.0, help="channel width multiplier")
    ap.add_argument("--limit_cpu_threads", type=int, default=24)
    ap.add_argument("--mixed_precision", action="store_true", help="enable mixed_float16 (optional)")
    args = ap.parse_args()

    if args.mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Using mixed_float16.")
        except Exception:
            print("Mixed precision not available; continuing in float32.")

    if args.limit_cpu_threads and args.limit_cpu_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(args.limit_cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, args.limit_cpu_threads // 2))


    train_ds, val_ds, num_classes = make_sequence_datasets(
        T=args.frames, batch_size=args.batch
    )


    model = build_3dcnn(num_classes=num_classes, T=args.frames, width_mult=args.width_mult)
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    if args.mixed_precision:
       
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")],
    )


    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=str(LOG_DIR), update_freq="batch"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CKPT_DIR / "ckpt_{epoch:02d}_valacc_{val_accuracy:.3f}.weights.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )


    final_weights = CKPT_DIR / "final.weights.h5"
    model.save_weights(final_weights)
    print(f"Saved final weights to: {final_weights}")

if __name__ == "__main__":
    main()
