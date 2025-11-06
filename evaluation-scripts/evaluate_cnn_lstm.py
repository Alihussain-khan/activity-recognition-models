from __future__ import annotations
import os, sys, json, re, glob
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt


ROOT = Path.home() / "bhome" / "group8"
LOG_DIR = ROOT / "logs" / "cnn_lstm_inception_v1_tuned"
CKPT_DIR = ROOT / "checkpoints" / "cnn_lstm_inception_v1_tuned"
OUT_DIR = ROOT / "metrics" / "cnn_lstm_inception_v1_tuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)


PROJECT_ROOT = ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_cnn_lstm(num_classes: int, T: int, train_backbone: bool = False) -> tf.keras.Model:
    frame_input = tf.keras.Input(shape=(T, 224, 224, 3), name="clip") 
    backbone = hub.KerasLayer(
        "https://tfhub.dev/google/imagenet/inception_v1/feature_vector/5",
        trainable=train_backbone,
        name="inception_v1_backbone",
    )
    td = tf.keras.layers.TimeDistributed(backbone, name="td_backbone")(frame_input) 
    x = tf.keras.layers.Dropout(0.3)(td)
    x = tf.keras.layers.LSTM(512, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    logits = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(frame_input, logits)


def _seq_from_video_tensor(video: tf.Tensor, T: int) -> tf.Tensor:

    t = tf.shape(video)[0]

    idxs = tf.cast(tf.linspace(0.0, tf.maximum(tf.cast(t, tf.float32) - 1.0, 0.0), T), tf.int32)
    frames = tf.gather(video, idxs)  
    frames = tf.image.resize(frames, (224, 224))
    frames = tf.cast(frames, tf.float32) / 255.0
    return frames 

def make_test_sequence_dataset(T: int, batch_size: int = 8):

    try:
        
        from src.data_ucf101_tfds import make_test_sequence_tfds 
        test_ds, num_classes = make_test_sequence_tfds(T=T, batch_size=batch_size)
        return test_ds, num_classes
    except Exception:
        import tensorflow_datasets as tfds
        ds = tfds.load("ucf101", split="test", as_supervised=False) 
        info = tfds.builder("ucf101").info
        num_classes = info.features["label"].num_classes

        def _prep(ex):
            clip = _seq_from_video_tensor(ex["video"], T=T)        
            label = tf.cast(ex["label"], tf.int32)
            return clip, label

        ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds, num_classes

_CKPT_RE = re.compile(r"valacc_(\d+\.\d+)")

def _auto_pick_checkpoint() -> Path | None:

    candidates = list(map(Path, glob.glob(str(CKPT_DIR / "ckpt_*_valacc_*.weights.h5"))))
    best = None
    best_acc = -1.0
    for p in candidates:
        m = _CKPT_RE.search(p.name)
        if not m:
            continue
        acc = float(m.group(1))
        if acc > best_acc:
            best_acc, best = acc, p
    if best is not None:
        return best
    final_p = CKPT_DIR / "final.weights.h5"
    return final_p if final_p.exists() else None

def confusion_matrix_png(y_true: np.ndarray, y_pred: np.ndarray, out_png: Path, num_classes: int):

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()
    fig = plt.figure(figsize=(7, 6))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix (test)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([]); ax.set_yticks([]) 
    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Evaluate CNN+LSTM (Inception v1 backbone) on UCF101 test split.")
    ap.add_argument("--frames", type=int, default=16, help="Sequence length T used during training.")
    ap.add_argument("--batch", type=int, default=8, help="Batch size for evaluation.")
    ap.add_argument("--ckpt", type=str, default="", help="Optional path to weights.h5; if empty, auto-pick best/final.")
    ap.add_argument("--gpu", type=str, default="4", help="CUDA_VISIBLE_DEVICES setting, e.g. '0' or '4'.")
    ap.add_argument("--limit_cpu_threads", type=int, default=24, help="Limit CPU threads (0 to disable).")
    ap.add_argument("--export_preds", type=str, default="", help="Optional CSV path for per-example predictions.")
    ap.add_argument("--no_png_cm", action="store_true", help="Skip saving confusion_matrix.png")
    args = ap.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.limit_cpu_threads and args.limit_cpu_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(args.limit_cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, args.limit_cpu_threads // 2))


    test_ds, num_classes = make_test_sequence_dataset(T=args.frames, batch_size=args.batch)


    model = build_cnn_lstm(num_classes=num_classes, T=args.frames, train_backbone=False)
    model.compile(  
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")],
    )


    ckpt_path = Path(args.ckpt) if args.ckpt else _auto_pick_checkpoint()
    if ckpt_path is None or not ckpt_path.exists():
        raise SystemExit(f"No checkpoint found. Checked: explicit path={args.ckpt!r} and {CKPT_DIR}")
    model.load_weights(str(ckpt_path))
    print(f"[eval] Loaded weights: {ckpt_path}")


    print("[eval] Running model.evaluate() on TEST split…")
    results = model.evaluate(test_ds, verbose=1, return_dict=True)
    top1 = float(results.get("accuracy", np.nan))
    top5 = float(results.get("top5", np.nan))
    print(f"[eval] test_top1={top1:.6f}  test_top5={top5:.6f}")


    y_true_all, y_pred_all = [], []
    if args.export_preds:
        import csv
        out_csv = Path(args.export_preds)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        f_csv = open(out_csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(f_csv)
        writer.writerow(["true", "pred_top1"])
    else:
        writer = None
        f_csv = None

    for bx, by in test_ds:
        probs = model.predict(bx, verbose=0)          
        y_pred = np.argmax(probs, axis=1)             
        y_true = by.numpy()
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        if writer is not None:
            for t, p in zip(y_true, y_pred):
                writer.writerow([int(t), int(p)])

    if f_csv is not None:
        f_csv.close()
        print(f"[eval] Wrote per-example predictions CSV → {out_csv}")

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)


    metrics = {
        "checkpoint": str(ckpt_path),
        "test_top1": top1,
        "test_top5": top5,
        "num_classes": int(num_classes),
        "num_examples": int(y_true.shape[0]),
        "frames_T": int(args.frames),
        "batch": int(args.batch),
    }
    with (OUT_DIR / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[ok] Wrote {OUT_DIR/'metrics.json'}")

    if not args.no_png_cm:
        cm_png = OUT_DIR / "confusion_matrix.png"
        confusion_matrix_png(y_true, y_pred, cm_png, num_classes)
        print(f"[ok] Wrote {cm_png}")

    try:
        tb_writer = tf.summary.create_file_writer(str(LOG_DIR / "eval"))
        with tb_writer.as_default():
            tf.summary.scalar("eval/test_top1", top1, step=0)
            tf.summary.scalar("eval/test_top5", top5, step=0)
        print(f"[ok] Wrote eval scalars to TensorBoard dir: {LOG_DIR / 'eval'}")
    except Exception as e:
        print(f"(non-fatal) TensorBoard logging failed: {e}")

if __name__ == "__main__":
    main()
