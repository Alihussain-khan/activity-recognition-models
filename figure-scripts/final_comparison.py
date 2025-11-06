from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/stud/u286092/bhome/group8")


SF_SUMMARY   = ROOT / "metrics/single_frame_inception_v1_headonly/metrics.json"
LSTM_SUMMARY = ROOT / "metrics/cnn_lstm_inception_v1_tuned/metrics.json"
I3D_SUMMARY  = ROOT / "metrics/i3d_pretrained/metrics.json"

OUT_DIR = ROOT / "report_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / "model_comparison_top1.png"

def read_test_top1(p: Path):
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        for k in ("test_top1", "test_accuracy", "accuracy", "top1"):
            if k in d and d[k] is not None:
                return float(d[k])
    except Exception:
        pass
    return None

labels = []
values = []

sf = read_test_top1(SF_SUMMARY)
if sf is not None:
    labels.append("Single-Frame")
    values.append(sf)

lstm = read_test_top1(LSTM_SUMMARY)
if lstm is not None:
    labels.append("CNN+LSTM")
    values.append(lstm)

i3d = read_test_top1(I3D_SUMMARY)
if i3d is not None:
    labels.append("I3D")
    values.append(i3d)

if not values:
    raise SystemExit("No metrics found. Make sure the three metrics.json files exist and contain test_top1.")

# Plot
plt.figure(figsize=(7.2, 4.6))
x = np.arange(len(values))
plt.bar(x, values)
plt.xticks(x, labels)
for i, v in enumerate(values):
    plt.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=10, weight="bold")
plt.ylim(0, max(0.75, max(values) * 1.1))
plt.title("Test Top-1 Accuracy – Model Comparison")
plt.ylabel("Accuracy")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.close()
print(f"[ok] {OUT_PNG}")
