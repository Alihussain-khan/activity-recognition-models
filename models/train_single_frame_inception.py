
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from pathlib import Path
import argparse
import tensorflow as tf
import tensorflow_hub as hub


from src.data_ucf101_tfds import make_datasets_single_tfds as make_datasets
print("Visible GPUs to TensorFlow:", tf.config.list_physical_devices('GPU'))


ROOT = Path.home() / "bhome" / "group8"
LOG_DIR = ROOT / "logs" / "single_frame_inception_v1_headonly"
CKPT_DIR = ROOT / "checkpoints" / "single_frame_inception_v1_headonly"


def build_model(num_classes: int, train_backbone: bool = False) -> tf.keras.Model:

    inputs = tf.keras.Input(shape=(224, 224, 3), name="image")
    backbone = hub.KerasLayer(
        "https://tfhub.dev/google/imagenet/inception_v1/feature_vector/5",
        trainable=train_backbone,
        name="inception_v1_backbone",
    )
    x = backbone(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="logits")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_backbone", action="store_true",
                        help="Unfreeze and fine-tune Inception backbone")
    parser.add_argument("--limit_cpu_threads", type=int, default=24,
                        help="Max CPU threads (per lab policy). Use 0 to skip limiting.")
    parser.add_argument("--gpu", type=str, default=None,
                        help="Override CUDA_VISIBLE_DEVICES value (e.g. '4')")
    args = parser.parse_args()


    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    if args.limit_cpu_threads and args.limit_cpu_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(args.limit_cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, args.limit_cpu_threads // 2))


    train_ds, val_ds, num_classes = make_datasets(batch_size=args.batch)


    model = build_model(num_classes=num_classes, train_backbone=args.train_backbone)


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")],
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
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CKPT_DIR / "ckpt_e{epoch:03d}.weights.h5"),
            save_best_only=False,
            save_weights_only=True,
            verbose=0,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
            mode="max",
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
