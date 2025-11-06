
from __future__ import annotations
from pathlib import Path
import sys, re, csv, json
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


tb_make_ndarray = None
try:
    from tensorboard.util.tensor_util import make_ndarray as tb_make_ndarray  
except Exception:
    try:
        from tensorflow.python.framework.tensor_util import MakeNdarray as tb_make_ndarray  
    except Exception:
        pass
if tb_make_ndarray is None:
    raise RuntimeError("Could not import make_ndarray from TensorBoard/TF.")

MODELS = {
        "sf": dict(
        log_root=Path("/home/stud/u286092/bhome/group8/logs/single_frame_inception_v1"),
        dump_dir=Path("/home/stud/u286092/bhome/group8/report_figs/sf_dump_tensors"),
        title="Single-Frame",
    ),
      "sf_headonly": dict(
        log_root=Path("/home/stud/u286092/bhome/group8/logs/single_frame_inception_v1_headonly"),
        dump_dir=Path("/home/stud/u286092/bhome/group8/report_figs/sf_dump_tensors_headonly"),
        title="Single-Frame_headonly",
    ),
    "lstm": dict(
        log_root=Path("/home/stud/u286092/bhome/group8/logs/cnn_lstm_inception_v1_tuned"),
        dump_dir=Path("/home/stud/u286092/bhome/group8/report_figs/lstm_dump_tensors_tuned"),
        title="CNN+LSTM",
    ),
    "i3d": dict(
        log_root=Path("/home/stud/u286092/bhome/group8/logs/i3d_baseline"),
        dump_dir=Path("/home/stud/u286092/bhome/group8/report_figs/i3d_dump_tensors"),
        title="I3D Baseline",
    ),
        "i3d_pretrained": dict(
        log_root=Path("/home/stud/u286092/bhome/group8/logs/i3d_pretrained"),
        dump_dir=Path("/home/stud/u286092/bhome/group8/report_figs/i3d_pretrained_dump_tensors"),
        title="I3D Pretrained",
    ),
}


_re_metric = re.compile(r"(loss|acc(uracy)?|top[_\- ]?\d+|k=\d+)", re.I)

def is_numeric_arr(arr: np.ndarray) -> bool:
    return arr is not None and arr.dtype.kind in ("f", "i", "u") and arr.size >= 1

def find_run(log_root: Path) -> Path | None:
    if not log_root.exists():
        return None

    candidates = []

    if any(f.is_file() and f.name.startswith("events.out.tfevents") for f in log_root.iterdir()):
        candidates.append(("root", log_root))

    for p in log_root.rglob("*"):
        if not p.is_dir():
            continue
        n = sum(1 for f in p.iterdir() if f.is_file() and f.name.startswith("events.out.tfevents"))
        if n:
            
            name = p.name.lower()
            if "validation" in name or "val" == name:
                pref = 0
            elif "eval" in name:
                pref = 1
            elif "train" in name:
                pref = 2
            else:
                pref = 3
            candidates.append(((pref, -n, -p.stat().st_mtime), p))
    if not candidates:
        return None
   
    scored = []
    for s, p in candidates:
        if isinstance(s, tuple):
            scored.append((s, p))
        else:
           
            scored.append(((9, 0, 0), p))
    scored.sort(key=lambda t: t[0])
    return scored[0][1]

def dump_tensors(run: Path, outdir: Path):
    ea = EventAccumulator(str(run), size_guidance={"tensors": 0, "scalars": 0})
    ea.Reload()
    all_tags = ea.Tags()
    tensor_tags = all_tags.get("tensors", [])
    scalar_tags = all_tags.get("scalars", [])
    cand_tensor = [t for t in tensor_tags if _re_metric.search(t)]
    cand_scalar = [t for t in scalar_tags if _re_metric.search(t)]
    if not cand_tensor and not cand_scalar:
        print("[WARN] no metric-like tags found in tensors or scalars."); return


    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    skipped_text = 0

    def gather_series(tag: str):
   
        if tag in scalar_tags:
            evs = ea.Scalars(tag)
            xs = [float(e.step) for e in evs]
            ys = [float(e.value) for e in evs]
            return xs, ys
        if tag in tensor_tags:
            events = ea.Tensors(tag)
            xs, ys = [], []
            for ev in events:
                arr = tb_make_ndarray(ev.tensor_proto)
                if is_numeric_arr(arr):
                    xs.append(float(ev.step))
                    ys.append(float(arr.reshape(-1)[0]))
            return xs, ys
        return [], []


    for tag in sorted(set(cand_scalar) | set(cand_tensor)):
        xs, ys = gather_series(tag)
        if not xs:
            continue
        order = np.argsort(xs)
        xs = np.array(xs)[order]; ys = np.array(ys, dtype=float)[order]
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", tag)
        csv_path = outdir / f"{safe}.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["step","value","tag"])
            for x,y in zip(xs,ys): w.writerow([int(x), float(y), tag])
        summary[tag] = dict(n=len(xs), min=float(np.nanmin(ys)), max=float(np.nanmax(ys)))
        print(f"[csv] {csv_path}  (n={summary[tag]['n']}, max={summary[tag]['max']:.3f})")


        with (outdir / "_dump_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        if skipped_text:
            print(f"[info] skipped {skipped_text} non-numeric tensor events (e.g., model JSON).")
        print(f"[ok] wrote {outdir / '_dump_summary.json'}")

if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "lstm"  
    if key not in MODELS: raise SystemExit("Usage: python dump_tb_tensors_to_csv.py [lstm|i3d]")
    cfg = MODELS[key]
    run = find_run(cfg["log_root"])
    if not run: raise SystemExit(f"No TB event files under {cfg['log_root']}")
    print(f"[info] using run: {run}")
    dump_tensors(run, cfg["dump_dir"])
