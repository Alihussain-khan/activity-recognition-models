from __future__ import annotations
from pathlib import Path
import sys, csv
import numpy as np
import matplotlib.pyplot as plt


CFG = {
    "sf_headonly": dict(
        dump=Path("/home/stud/u286092/bhome/group8/report_figs/sf_dump_tensors_headonly"),
        out=Path("/home/stud/u286092/bhome/group8/report_figs/sf_report_clean"),
        title="Single-Frame_headonly",
    ),
    "lstm": dict(
        dump=Path("/home/stud/u286092/bhome/group8/report_figs/lstm_dump_tensors_tuned"),
        out=Path("/home/stud/u286092/bhome/group8/report_figs/lstm_report_clean_tuned"),
        title="CNN+LSTM",
    ),
    "i3d": dict(
        dump=Path("/home/stud/u286092/bhome/group8/report_figs/i3d_dump_tensors"),
        out=Path("/home/stud/u286092/bhome/group8/report_figs/i3d_report_clean"),
        title="I3D Baseline",
    ),
    "i3d_pretrained": dict(
        dump=Path("/home/stud/u286092/bhome/group8/report_figs/i3d_pretrained_dump_tensors"),
        out=Path("/home/stud/u286092/bhome/group8/report_figs/i3d_pretrained_report_clean"),
        title="I3D Pretrained",
    ),
}


def load_csv(p: Path):
    xs, ys = [], []
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row["step"]))
            ys.append(float(row["value"]))
    xs, ys = np.array(xs), np.array(ys)
    order = np.argsort(xs)
    return xs[order], ys[order]

def pick(files, keyword: str):

    kw = keyword.lower()
    for f in sorted(files):
        if kw in f.name.lower():
            return f
    return None


def smooth(y, k=3):
    if len(y) < 3 or k <= 1:
        return y
    return np.convolve(y, np.ones(k) / k, mode="same")

def annotate_best(ax, x, y, label, mode="max"):
    if len(y) == 0:
        return
    i = np.nanargmax(y) if mode == "max" else np.nanargmin(y)
    ax.scatter([x[i]], [y[i]], s=35)
    ax.annotate(f"{label}: {y[i]:.3f} @ {int(x[i])}",
                (x[i], y[i]), textcoords="offset points",
                xytext=(6, 6), fontsize=9, weight="bold")

def plot_metric(ax, x, y, title, ylabel, label, invert_best=False):
    if x is None or len(x) == 0:
        return
    ax.plot(x, smooth(y), label=label, linewidth=1.8)
    annotate_best(ax, x, y, label, mode=("min" if invert_best else "max"))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

def get_metric_key(data: dict, group: str, candidates: list[str]):

    for name in candidates:
        key = f"{group}_{name}"
        if key in data:
            return key, name
    return None, None


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "i3d_pretrained"
    if key not in CFG:
        keys = ", ".join(sorted(CFG.keys()))
        raise SystemExit(f"Usage: python make_figs_from_dump_cleansplit.py [{keys}]")

    dump = CFG[key]["dump"]
    out = CFG[key]["out"]
    title = CFG[key]["title"]

    out.mkdir(parents=True, exist_ok=True)
    files = list(dump.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files in {dump}")


    epoch_files = [f for f in files if f.name.startswith("epoch_")]
    eval_files  = [f for f in files if f.name.startswith("evaluation_") or f.name.startswith("eval_")]


    data = {}
    for group, flist in [("epoch", epoch_files), ("eval", eval_files)]:
        for metric in ["top1", "accuracy", "top5", "loss"]:
            f = pick(flist, metric)
            if f:
                data[f"{group}_{metric}"] = load_csv(f)


    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=False)
    fig.suptitle(f"{title} – Training (Epoch-level)", fontsize=13)


    key_metric, picked = get_metric_key(data, "epoch", ["top1", "accuracy"])
    if key_metric:
        x, y = data[key_metric]
        panel_title = "Top-1 Accuracy vs Epoch" if picked == "top1" else "Accuracy vs Epoch"
        ylabel = "Top-1 Accuracy" if picked == "top1" else "Accuracy"
        plot_metric(axes[0], x, y, panel_title, ylabel, "train")


    if "epoch_top5" in data:
        x, y = data["epoch_top5"]
        plot_metric(axes[1], x, y, "Top-5 Accuracy vs Epoch", "Top-5 Accuracy", "train")


    if "epoch_loss" in data:
        x, y = data["epoch_loss"]
        plot_metric(axes[2], x, y, "Loss vs Epoch", "Loss", "train", invert_best=True)
        axes[2].set_xlabel("Epoch")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    p = out / "train_epoch_metrics.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"[ok] {p}")

 
    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=False)
    fig.suptitle(f"{title} – Evaluation (Per-iteration)", fontsize=13)

    key_metric, picked = get_metric_key(data, "eval", ["top1", "accuracy"])
    if key_metric:
        x, y = data[key_metric]
        panel_title = "Validation Top-1 Accuracy" if picked == "top1" else "Validation Accuracy"
        ylabel = "Top-1 Accuracy" if picked == "top1" else "Accuracy"
        plot_metric(axes[0], x, y, panel_title, ylabel, "eval")

    if "eval_top5" in data:
        x, y = data["eval_top5"]
        plot_metric(axes[1], x, y, "Validation Top-5 Accuracy", "Top-5 Accuracy", "eval")

    if "eval_loss" in data:
        x, y = data["eval_loss"]
        plot_metric(axes[2], x, y, "Validation Loss", "Loss", "eval", invert_best=True)
        axes[2].set_xlabel("Step")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    p = out / "eval_iteration_metrics.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"[ok] {p}")

if __name__ == "__main__":
    main()
