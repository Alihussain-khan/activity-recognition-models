from __future__ import annotations
import os, json
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt


ROOT = Path.home() / "bhome" / "group8"
LOG_DIR = ROOT / "logs" / "single_frame_inception_v1_headonly"
CKPT_DIR = ROOT / "checkpoints" / "single_frame_inception_v1_headonly"
OUT_DIR = ROOT / "metrics" / "single_frame_inception_v1_headonly"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_model(num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image")
    backbone = hub.KerasLayer(
        "https://tfhub.dev/google/imagenet/inception_v1/feature_vector/5",
        trainable=False,
        name="inception_v1_backbone",
    )
    x = backbone(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="logits")(x)
    return tf.keras.Model(inputs, outputs)


def make_test_dataset(batch_size: int = 32):
    try:
        from src.data_ucf101_tfds import make_test_single_tfds  
        test_ds, num_classes = make_test_single_tfds(batch_size=batch_size)
        return test_ds, num_classes
    except Exception:

        import tensorflow_datasets as tfds
        builder = tfds.builder("ucf101")
        builder.download_and_prepare()  
        info = builder.info
        num_classes = info.features["label"].num_classes
        def _prep(ex):

            vid = ex["video"] 
            t = tf.shape(vid)[0]
            idx = tf.maximum(0, t // 2)
            frame = vid[idx]
            frame = tf.image.resize(frame, (224, 224))
            frame = tf.cast(frame, tf.float32) / 255.0
            return frame, ex["label"]
        ds = tfds.load("ucf101", split="test", as_supervised=False)
        ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds, num_classes

def topk_acc(y_true, probs, k=5):
    topk = np.argpartition(-probs, kth=k-1, axis=1)[:, :k]
    correct = np.any(topk == y_true[:, None], axis=1)
    return float(np.mean(correct))

def confusion_matrix_png(y_true, y_pred, out_png: Path, num_classes: int):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ckpt", type=str, default="", help="Optional path to weights.h5; if empty, auto-pick best/final")
    args = ap.parse_args()


    test_ds, num_classes = make_test_dataset(batch_size=args.batch)

    model = build_model(num_classes)
    model.compile(  
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")],
    )


    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if not ckpt_path:
        cand_best = CKPT_DIR / "best_val_top1.weights.h5"
        cand_final = CKPT_DIR / "final.weights.h5"
        ckpt_path = cand_best if cand_best.exists() else cand_final
    if not ckpt_path.exists():
        raise SystemExit(f"No checkpoint found at {ckpt_path}.")
    model.load_weights(ckpt_path)
    print(f"[eval] loaded weights: {ckpt_path}")

    results = model.evaluate(test_ds, verbose=1, return_dict=True)
    top1 = float(results.get("accuracy", np.nan))
    top5 = float(results.get("top5", np.nan))

    y_true_all, y_pred_all = [], []
    for bx, by in test_ds:
        probs = model.predict(bx, verbose=0)
        y_true_all.append(by.numpy())
        y_pred_all.append(np.argmax(probs, axis=1))
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    metrics = {
        "checkpoint": str(ckpt_path),
        "test_top1": top1,
        "test_top5": top5,
        "num_classes": num_classes,
        "num_examples": int(y_true.shape[0]),
    }
    with (OUT_DIR / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    confusion_matrix_png(y_true, y_pred, OUT_DIR / "confusion_matrix.png", num_classes)
    print(f"[ok] wrote {OUT_DIR/'metrics.json'} and {OUT_DIR/'confusion_matrix.png'}")

if __name__ == "__main__":
    main()
