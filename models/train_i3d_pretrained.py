from __future__ import annotations
import os, sys, math, json, random
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

ROOT = Path.home() / "bhome" / "group8"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_ucf101_tfds import make_sequence_datasets_tfds as make_sequence_datasets

LOG_DIR  = ROOT / "logs" / "i3d_pretrained"
CKPT_DIR = ROOT / "checkpoints" / "i3d_pretrained"


def safe_mkdir(p: Path): p.mkdir(parents=True, exist_ok=True)

def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def get_cardinality(ds):
    try:
        c = int(tf.data.experimental.cardinality(ds).numpy())
        return c if 0 < c < 1e7 else None
    except Exception:
        return None


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps=1000, alpha=0.1):
        super().__init__()
        self.base_lr = tf.convert_to_tensor(base_lr, dtype=tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = self.base_lr * (step / tf.maximum(1.0, self.warmup_steps))
        denom = tf.maximum(1.0, (self.total_steps - self.warmup_steps))
        progress = tf.clip_by_value((step - self.warmup_steps) / denom, 0.0, 1.0)
        cosine = 0.5 * (1.0 + tf.cos(tf.constant(math.pi) * progress))
        decayed = (self.alpha + (1.0 - self.alpha) * cosine) * self.base_lr
        return tf.where(step < self.warmup_steps, warm, decayed)

def make_adamw(lr_sched, weight_decay=3e-4, clip_norm=1.0):
    return tf.keras.optimizers.AdamW(
        learning_rate=lr_sched,
        weight_decay=weight_decay,
        global_clipnorm=clip_norm,
        beta_1=0.9, beta_2=0.999
    )

def make_sparse_cce(num_classes, label_smoothing=0.1):
    ce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    @tf.function
    def loss_fn(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        y_oh = tf.one_hot(y_true, depth=num_classes)
        return ce(y_oh, y_pred)
    return loss_fn


def build_i3d_pretrained(num_classes: int, T: int, trainable_backbone=False) -> tf.keras.Model:
    handle = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"

    inputs = tf.keras.Input(shape=(T, 224, 224, 3), dtype=tf.float32, name="clip")
    hub_layer = hub.KerasLayer(handle, trainable=trainable_backbone, name="i3d_backbone")


    features = hub_layer(inputs)           
    x = tf.keras.layers.Dense(512, activation="relu")(features)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    return tf.keras.Model(inputs, outputs, name="i3d_pretrained")


class PeriodicCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, outdir: Path, every: int = 2):
        super().__init__()
        self.outdir = outdir; self.every = every
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every == 0:
            p = self.outdir / f"ckpt_e{epoch+1:03d}.weights.h5"
            self.model.save_weights(p)
            print(f"[ckpt] saved periodic weights -> {p}")
        self.model.save_weights(self.outdir / "ckpt_last.weights.h5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--base_lr", type=float, default=3e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--limit_cpu_threads", type=int, default=24)
    ap.add_argument("--mixed_precision", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trainable_backbone", action="store_true")
    args = ap.parse_args()

    if args.mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Using mixed precision.")
        except Exception:
            print("Mixed precision not available; using float32.")

    set_seeds(args.seed)

    if args.limit_cpu_threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.limit_cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, args.limit_cpu_threads // 2))


    train_ds, val_ds, num_classes = make_sequence_datasets(T=args.frames, batch_size=args.batch)

    steps_per_epoch = get_cardinality(train_ds) or 1000
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(steps_per_epoch, 500)


    print(f"[info] Loading pretrained Inflated Inception-v1 (I3D) backbone, trainable_backbone={args.trainable_backbone}")
    model = build_i3d_pretrained(num_classes, args.frames, args.trainable_backbone)


    lr_sched = WarmupCosine(args.base_lr, total_steps, warmup_steps, alpha=0.1)
    opt = make_adamw(lr_sched, weight_decay=3e-4)
    loss = make_sparse_cce(num_classes, args.label_smoothing)

    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="top1"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
    ]
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    safe_mkdir(LOG_DIR); safe_mkdir(CKPT_DIR)

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=str(LOG_DIR)),
        tf.keras.callbacks.CSVLogger(str(LOG_DIR / "training_log.csv"), append=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CKPT_DIR / "best_top1_val{val_top1:.3f}_epoch{epoch:03d}.weights.h5"),
            monitor="val_top1", mode="max", save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CKPT_DIR / "best_top5_val{val_top5:.3f}_epoch{epoch:03d}.weights.h5"),
            monitor="val_top5", mode="max", save_best_only=True, save_weights_only=True),
        PeriodicCheckpoint(CKPT_DIR, every=2),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_top5", patience=12, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.BackupAndRestore(str(CKPT_DIR / "backup")),
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

    summary = {
        "frames": args.frames,
        "batch": args.batch,
        "epochs": args.epochs,
        "base_lr": args.base_lr,
        "label_smoothing": args.label_smoothing,
        "trainable_backbone": args.trainable_backbone,
        "steps_per_epoch": int(steps_per_epoch),
        "warmup_steps": int(warmup_steps),
    }
    with (LOG_DIR / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] wrote {LOG_DIR/'run_summary.json'}")

if __name__ == "__main__":
    main()
