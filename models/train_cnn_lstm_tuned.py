from __future__ import annotations
import os
from pathlib import Path
import argparse
import tensorflow as tf
import tensorflow_hub as hub


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4") 
ROOT = Path.home() / "bhome" / "group8"
LOG_DIR = ROOT / "logs" / "cnn_lstm_inception_v1_tuned"
CKPT_DIR = ROOT / "checkpoints" / "cnn_lstm_inception_v1_tuned"


def build_cnn_lstm(
    num_classes: int,
    T: int,
    train_backbone: bool = False,
    lstm_units: int = 512,
    bidirectional: bool = False,
    dropout_td: float = 0.3,
    dropout_head: float = 0.3,
) -> tf.keras.Model:
    frame_input = tf.keras.Input(shape=(T, 224, 224, 3), name="clip")
    backbone = hub.KerasLayer(
        "https://tfhub.dev/google/imagenet/inception_v1/feature_vector/5",
        trainable=train_backbone,
        name="inception_v1_backbone",
    )
    x = tf.keras.layers.TimeDistributed(backbone, name="td_backbone")(frame_input) 
    x = tf.keras.layers.Dropout(dropout_td, name="dropout_td")(x)

    if bidirectional:
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units, return_sequences=False),
            name="bilstm",
        )(x)
    else:
        x = tf.keras.layers.LSTM(lstm_units, return_sequences=False, name="lstm")(x)

    x = tf.keras.layers.Dropout(dropout_head, name="dropout_head")(x)
    logits = tf.keras.layers.Dense(num_classes, activation="softmax", name="logits")(x)
    return tf.keras.Model(frame_input, logits, name="cnn_lstm_inception_v1")

def train(
    frames: int,
    batch: int,
    epochs: int,
    lr: float,
    label_smoothing: float,
    limit_cpu_threads: int,
    train_backbone: bool,
    fine_tune_after: int,
    lstm_units: int,
    bidirectional: bool,
    dropout_td: float,
    dropout_head: float,
    mixed_precision: bool,
    weight_decay: float,
    clip_norm: float,
):

    if limit_cpu_threads and limit_cpu_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(limit_cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, limit_cpu_threads // 2))

    if mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("[mp] Using mixed precision (float16).")
        except Exception as e:
            print(f"[mp] Mixed precision not available: {e}")

 
    from src.data_ucf101_tfds import make_sequence_datasets_tfds as make_sequence_datasets
    train_ds, val_ds, num_classes = make_sequence_datasets(T=frames, batch_size=batch)


    model = build_cnn_lstm(
        num_classes=num_classes,
        T=frames,
        train_backbone=train_backbone if fine_tune_after <= 0 else False,  
        lstm_units=lstm_units,
        bidirectional=bidirectional,
        dropout_td=dropout_td,
        dropout_head=dropout_head,
    )

    if weight_decay > 0:
        opt = tf.keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
            global_clipnorm=clip_norm if clip_norm > 0 else None,
            beta_1=0.9, beta_2=0.999,
        )
    else:
        opt = tf.keras.optimizers.Adam(
            learning_rate=lr,
            global_clipnorm=clip_norm if clip_norm > 0 else None,
        )


    loss = make_sparse_cce(num_classes, label_smoothing=label_smoothing)
    metrics = [
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
    ]

    model.compile(optimizer=opt, loss=loss, metrics=metrics)


    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=str(LOG_DIR), update_freq="batch"),
        tf.keras.callbacks.CSVLogger(str(LOG_DIR / "training_log.csv"), append=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CKPT_DIR / "best_valacc_{val_accuracy:.3f}_epoch{epoch:03d}.weights.h5"),
            monitor="val_accuracy",
            mode="max",
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
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-6, mode="max", verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1
        ),
    ]

    if fine_tune_after > 0:

        warm_epochs = min(fine_tune_after, epochs)
        remain = max(0, epochs - warm_epochs)
        print(f"[train] Phase 1: frozen backbone for {warm_epochs} epochs.")
        model.fit(train_ds, validation_data=val_ds, epochs=warm_epochs, callbacks=callbacks, verbose=1)

        if remain > 0:
            print("[train] Unfreezing backbone for fine-tuning…")
            for layer in model.layers:
                if "inception_v1_backbone" in layer.name:
                    layer.trainable = True

            model.compile(optimizer=opt, loss=loss, metrics=metrics)
            print(f"[train] Phase 2: fine-tune for {remain} epochs.")
            model.fit(train_ds, validation_data=val_ds, epochs=remain, callbacks=callbacks, verbose=1)
    else:
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)


    final_weights = CKPT_DIR / "final.weights.h5"
    model.save_weights(final_weights)
    print(f"Saved final weights to: {final_weights}")

def make_sparse_cce(num_classes: int, label_smoothing: float = 0.0):
    base = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    @tf.function
    def loss_fn(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_oh = tf.one_hot(y_true, depth=num_classes)
        return base(y_oh, y_pred)
    return loss_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=16, help="sequence length T")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--limit_cpu_threads", type=int, default=24)
    ap.add_argument("--train_backbone", action="store_true", help="Train TFHub backbone from the start")
    ap.add_argument("--fine_tune_after", type=int, default=6, help="Unfreeze backbone after N warmup epochs (0=disabled)")
    ap.add_argument("--lstm_units", type=int, default=512)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--dropout_td", type=float, default=0.3)
    ap.add_argument("--dropout_head", type=float, default=0.3)
    ap.add_argument("--mixed_precision", action="store_true")
    ap.add_argument("--weight_decay", type=float, default=3e-4)
    ap.add_argument("--clip_norm", type=float, default=1.0)
    args = ap.parse_args()

    train(
        frames=args.frames,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        label_smoothing=args.label_smoothing,
        limit_cpu_threads=args.limit_cpu_threads,
        train_backbone=args.train_backbone,
        fine_tune_after=args.fine_tune_after,
        lstm_units=args.lstm_units,
        bidirectional=args.bidirectional,
        dropout_td=args.dropout_td,
        dropout_head=args.dropout_head,
        mixed_precision=args.mixed_precision,
        weight_decay=args.weight_decay,
        clip_norm=args.clip_norm,
    )

if __name__ == "__main__":
    main()
