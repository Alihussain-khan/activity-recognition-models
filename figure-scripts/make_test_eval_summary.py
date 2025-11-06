
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


MODELS = {
    "sf_headonly": dict(
        title="Single-Frame (head-only)",
        metrics_dir=Path("/home/stud/u286092/bhome/group8/metrics/single_frame_inception_v1_headonly"),
        out_dir=Path("/home/stud/u286092/bhome/group8/report_figs/sf_headonly_test_report"),
    ),
    "lstm": dict(
        title="CNN + LSTM",
        metrics_dir=Path("/home/stud/u286092/bhome/group8/metrics/cnn_lstm_inception_v1_tuned"),
        out_dir=Path("/home/stud/u286092/bhome/group8/report_figs/lstm_test_report_tuned"),
    ),
    "i3d": dict(
        title="I3D Baseline",
        metrics_dir=Path("/home/stud/u286092/bhome/group8/metrics/i3d_baseline"),
        out_dir=Path("/home/stud/u286092/bhome/group8/report_figs/i3d_test_report"),
    ),
    "i3d_pretrained": dict(
        title="I3D Pretrained",
        metrics_dir=Path("/home/stud/u286092/bhome/group8/metrics/i3d_pretrained"),
        out_dir=Path("/home/stud/u286092/bhome/group8/report_figs/i3d_pretrained_test_report"),
    ),
}


def _first(d: dict, keys: list[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _fmt_percent(v):
    return f"{100.0 * v:.1f}%" if v is not None else "–"

def _load_image(path: Path):
    try:
        return Image.open(path)
    except Exception:
        return None


def plot_test_summary(model_key: str):
    if model_key not in MODELS:
        print(f"[error] unknown key '{model_key}'. Valid: {', '.join(MODELS)}")
        return

    cfg = MODELS[model_key]
    title = cfg["title"]
    mdir: Path = cfg["metrics_dir"]
    outdir: Path = cfg["out_dir"]
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_path = mdir / "metrics.json"
    cm_path = mdir / "confusion_matrix.png"

    if not metrics_path.exists():
        print(f"[warn] {metrics_path} missing → skip {model_key}")
        return

    with metrics_path.open() as f:
        metrics = json.load(f)


    top1 = _as_float(_first(metrics, ["test_top1", "top1", "accuracy"]))
    top5 = _as_float(_first(metrics, ["test_top5", "top5"]))
    loss = _as_float(_first(metrics, ["test_loss", "loss"]))

    n = _first(metrics, ["num_examples"])
    C = _first(metrics, ["num_classes"])
    ckpt = _first(metrics, ["checkpoint"])
    frames_T = _first(metrics, ["frames_T", "frames"])
    batch = _first(metrics, ["batch"])

    has_cm = cm_path.exists()
    if has_cm:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.1, 1.0]})
        ax_bar, ax_cm = axes
    else:
        fig, ax_bar = plt.subplots(1, 1, figsize=(6.5, 4.6))
        ax_cm = None

    fig.suptitle(f"{title} — Test Evaluation", fontsize=13, weight="bold", y=0.98)


    labels, vals, is_pct = [], [], []
    if top1 is not None: labels.append("Top-1"); vals.append(top1); is_pct.append(True)
    if top5 is not None: labels.append("Top-5"); vals.append(top5); is_pct.append(True)
    if loss is not None: labels.append("Loss");  vals.append(loss); is_pct.append(False)

    if vals:
        x = np.arange(len(vals))
        bars = ax_bar.bar(x, vals, width=0.55)
        ax_bar.set_xticks(x, labels)

        if any(is_pct):

            ymax = max(vals) if not all(is_pct) else max(1.0, max(vals) * 1.05)
            ax_bar.set_ylim(0, ymax)
        ax_bar.set_ylabel("Value")
        ax_bar.grid(axis="y", alpha=0.3)
        for i, (b, v, p) in enumerate(zip(bars, vals, is_pct)):
            txt = _fmt_percent(v) if p else f"{v:.3f}"
            ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, txt,
                        ha="center", va="bottom", fontsize=9, weight="bold")
        ax_bar.set_title("Overall Test Metrics", fontsize=11)
    else:
        ax_bar.set_title("No metrics found in metrics.json", fontsize=11)


    meta = []
    if n: meta.append(f"N={n}")
    if C: meta.append(f"Classes={C}")
    if frames_T: meta.append(f"T={frames_T}")
    if batch: meta.append(f"Batch={batch}")
    if ckpt: meta.append(f"Ckpt: {Path(ckpt).name}")
    if meta:
        fig.text(0.01, 0.01, " • ".join(meta), fontsize=9, color="#333")

    if ax_cm is not None:
        img = _load_image(cm_path)
        if img is not None:
            ax_cm.imshow(img)
            ax_cm.set_axis_off()
            ax_cm.set_title("Confusion Matrix (Test)", fontsize=11)
        else:
            ax_cm.text(0.5, 0.5, "Could not load confusion_matrix.png",
                       ha="center", va="center")
            ax_cm.set_axis_off()


    out_png = outdir / "test_summary.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_png, dpi=250)
    plt.close()
    print(f"[ok] wrote {out_png}")

if __name__ == "__main__":
    import sys
    key = sys.argv[1] if len(sys.argv) > 1 else "sf_headonly"
    plot_test_summary(key)
