
from __future__ import annotations
import os, sys, re, glob, json
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt


ROOT = Path.home() / "bhome" / "group8"
LOG_DIR  = ROOT / "logs" / "i3d_pretrained"
CKPT_DIR = ROOT / "checkpoints" / "i3d_pretrained"
OUT_DIR  = ROOT / "metrics" / "i3d_pretrained"
OUT_DIR.mkdir(parents=True, exist_ok=True)


if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))


def _seq_from_video_tensor(video: tf.Tensor, T: int) -> tf.Tensor:
    t = tf.shape(video)[0]
    idxs = tf.cast(tf.linspace(0.0, tf.maximum(tf.cast(t, tf.float32) - 1.0, 0.0), T), tf.int32)
    frames = tf.gather(video, idxs)             
    frames = tf.image.resize(frames, (224, 224)) 
    frames = tf.cast(frames, tf.float32) / 255.0
    return frames

def make_test_sequence_dataset(T: int, batch_size: int):
    try:
      
        from src.data_ucf101_tfds import make_test_sequence_tfds  
        return make_test_sequence_tfds(T=T, batch_size=batch_size)
    except Exception:
        import tensorflow_datasets as tfds
        ds = tfds.load("ucf101", split="test", as_supervised=False)
        info = tfds.builder("ucf101").info
        num_classes = info.features["label"].num_classes
        def _prep(ex):
            clip = _seq_from_video_tensor(ex["video"], T)
            label = tf.cast(ex["label"], tf.int32)
            return clip, label
        ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds, num_classes


def build_i3d_pretrained(num_classes: int, T: int, trainable_backbone: bool = False) -> tf.keras.Model:
    handle = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
    inputs = tf.keras.Input(shape=(T, 224, 224, 3), dtype=tf.float32, name="clip")
    hub_layer = hub.KerasLayer(handle, trainable=trainable_backbone, name="i3d_backbone")
    features = hub_layer(inputs)              
    x = tf.keras.layers.Dense(512, activation="relu")(features)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inputs, outputs, name="i3d_pretrained")


def pick_checkpoint() -> Path | None:
    pats = [
        "best_top1_val*.weights.h5",
        "best_top5_val*.weights.h5",
        "ckpt_e*.weights.h5",
        "ckpt_last.weights.h5",
        "final.weights.h5",
    ]
    for pat in pats:
        cands = [Path(p) for p in glob.glob(str(CKPT_DIR / pat))]
        if not cands: continue
       
        m = re.search(r"val(\d+\.\d+)", pat)
        if m:
            def score(p):
                m2 = re.search(r"val(\d+\.\d+)", p.name)
                return float(m2.group(1)) if m2 else -1.0
            cands.sort(key=score, reverse=True)
        else:
            cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0]
    return None

def confusion_matrix_png(y_true, y_pred, out_png: Path, num_classes: int):
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
    ap = argparse.ArgumentParser(description="Evaluate i3d_pretrained (UCF101 test).")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--gpu", type=str, default="4")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--export_preds", type=str, default="")
    ap.add_argument("--limit_cpu_threads", type=int, default=24)
    ap.add_argument("--no_png_cm", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.limit_cpu_threads and args.limit_cpu_threads > 0:
        tf.config.threading.set_intra_op_parallelism_threads(args.limit_cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, args.limit_cpu_threads // 2))

    test_ds, num_classes = make_test_sequence_dataset(T=args.frames, batch_size=args.batch)

    model = build_i3d_pretrained(num_classes=num_classes, T=args.frames, trainable_backbone=False)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="top1"),
                           tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")])

    ckpt = Path(args.ckpt) if args.ckpt else pick_checkpoint()
    if ckpt is None or not ckpt.exists():
        raise SystemExit(f"No checkpoint found. Looked in {CKPT_DIR}")
    model.load_weights(str(ckpt))
    print(f"[eval] Loaded: {ckpt}")

    print("[eval] Evaluating on TEST…")
    res = model.evaluate(test_ds, verbose=1, return_dict=True)
    top1 = float(res.get("top1", np.nan))
    top5 = float(res.get("top5", np.nan))
    print(f"[eval] test_top1={top1:.6f}  test_top5={top5:.6f}")


    writer = None; f_csv = None
    if args.export_preds:
        import csv
        out_csv = Path(args.export_preds)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        f_csv = open(out_csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(f_csv); writer.writerow(["true","pred_top1"])

    y_true_all, y_pred_all = [], []
    for bx, by in test_ds:
        probs = model.predict(bx, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        y_true = by.numpy()
        y_true_all.append(y_true); y_pred_all.append(y_pred)
        if writer:
            for t, p in zip(y_true, y_pred): writer.writerow([int(t), int(p)])

    if f_csv: f_csv.close()

    y_true = np.concatenate(y_true_all); y_pred = np.concatenate(y_pred_all)
    (OUT_DIR).mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "metrics.json").open("w") as f:
        json.dump(dict(
            checkpoint=str(ckpt),
            test_top1=top1, test_top5=top5,
            num_classes=int(num_classes),
            num_examples=int(y_true.shape[0]),
            frames_T=int(args.frames), batch=int(args.batch)
        ), f, indent=2)
    print(f"[ok] wrote {OUT_DIR/'metrics.json'}")

    if not args.no_png_cm:
        cm_png = OUT_DIR / "confusion_matrix.png"
        confusion_matrix_png(y_true, y_pred, cm_png, num_classes)
        print(f"[ok] wrote {cm_png}")


    try:
        tbw = tf.summary.create_file_writer(str(LOG_DIR / "eval"))
        with tbw.as_default():
            tf.summary.scalar("eval/test_top1", top1, step=0)
            tf.summary.scalar("eval/test_top5", top5, step=0)
        print(f"[ok] TB scalars -> {LOG_DIR/'eval'}")
    except Exception as e:
        print(f"(non-fatal) TB logging failed: {e}")

if __name__ == "__main__":
    main()
