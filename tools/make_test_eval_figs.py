
from __future__ import annotations
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


MODELS = {
    "sf_headonly": dict(
        title="Single-Frame_headonly",
        metrics_dir=Path("/home/stud/u286092/bhome/group8/metrics/single_frame_inception_v1_headonly"),
        out_dir=Path("/home/stud/u286092/bhome/group8/report_figs/sf_headonly_test_report"),
    ),
    "lstm": dict(
        title="CNN+LSTM",
        metrics_dir=Path("/home/stud/u286092/bhome/group8/metrics/cnn_lstm_inception_v1"),
        out_dir=Path("/home/stud/u286092/bhome/group8/report_figs/lstm_test_report"),
    ),
    "i3d": dict(
        title="I3D Baseline",
        metrics_dir=Path("/home/stud/u286092/bhome/group8/metrics/i3d_baseline"),
        out_dir=Path("/home/stud/u286092/bhome/group8/report_figs/i3d_test_report"),
    ),
}



def plot_test_summary(model_key: str):
    cfg = MODELS[model_key]
    title = cfg["title"]
    mdir = cfg["metrics_dir"]
    outdir = cfg["out_dir"]
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_path = mdir / "metrics.json"
    cm_path = mdir / "confusion_matrix.png"


    if not metrics_path.exists():
        print(f"[warn] metrics.json missing for {model_key}")
        return
    with open(metrics_path) as f:
        metrics = json.load(f)


    loss = metrics.get("test_loss") or metrics.get("loss")
    top1 = metrics.get("test_top1") or metrics.get("accuracy") or metrics.get("top1")
    top5 = metrics.get("test_top5") or metrics.get("top5")

    num_examples = metrics.get("num_examples")
    num_classes = metrics.get("num_classes")
    class_acc = metrics.get("class_accuracy")


    fig_height = 6 if class_acc else 4
    fig, axes = plt.subplots(2 if class_acc else 1, 1, figsize=(6, fig_height))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    fig.suptitle(f"{title} – Test Evaluation Summary", fontsize=13, weight="bold", y=0.98)


    labels, vals = [], []
    if top1 is not None:
        labels.append("Top-1 Accuracy")
        vals.append(top1)
    if top5 is not None:
        labels.append("Top-5 Accuracy")
        vals.append(top5)
    if loss is not None:
        labels.append("Loss")
        vals.append(loss)

    ax = axes[0]
    colors = ["#2a6fdb", "#28a745", "#cc3300"][: len(vals)]
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9, weight="bold")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)


    if num_examples or num_classes:
        info = []
        if num_examples:
            info.append(f"N={num_examples}")
        if num_classes:
            info.append(f"Classes={num_classes}")
        ax.set_title("Overall Test Metrics  (" + ", ".join(info) + ")", fontsize=11)
    else:
        ax.set_title("Overall Test Metrics", fontsize=11)


    if class_acc:
        names = list(class_acc.keys())
        accs = np.array(list(class_acc.values()))
        order = np.argsort(accs)
        names = np.array(names)[order]
        accs = accs[order]
        ax = axes[1]
        ax.barh(names, accs, color="#007acc")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Accuracy")
        ax.set_title("Per-class Test Accuracy", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.96])


    out_png = outdir / "test_summary_metrics.png"
    plt.savefig(out_png, dpi=250)
    plt.close()
    print(f"[ok] wrote {out_png}")


    if cm_path.exists():
        img = Image.open(cm_path)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img)
        ax.set_title(f"{title} – Confusion Matrix (Test)")
        ax.axis("off")
        plt.tight_layout()
        out_cm = outdir / "test_confusion_matrix.png"
        plt.savefig(out_cm, dpi=250)
        plt.close()
        print(f"[ok] wrote {out_cm}")


if __name__ == "__main__":
    import sys

    key = sys.argv[1] if len(sys.argv) > 1 else "sf_headonly"
    if key not in MODELS:
        print("Usage: python make_test_eval_summary.py [sf_headonly|lstm|i3d]")
    else:
        plot_test_summary(key)
