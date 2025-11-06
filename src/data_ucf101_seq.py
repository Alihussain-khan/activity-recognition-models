
from __future__ import annotations
import os, cv2, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Iterable, Dict
import numpy as np
import tensorflow as tf

ROOT = Path.home() / "bhome" / "group8"
MANUAL = ROOT / "tensorflow_datasets" / "downloads" / "manual"
UCF_VIDEOS = MANUAL / "UCF-101"
UCF_SPLITS = MANUAL / "ucfTrainTestlist"
CLASS_IND = UCF_SPLITS / "classInd.txt"
TRAIN_SPLIT = UCF_SPLITS / "trainlist01.txt"
TEST_SPLIT  = UCF_SPLITS / "testlist01.txt"

IMG_SIZE = (224, 224)

@dataclass
class VideoItem:
    path: str
    label: int

def _read_class_map() -> Dict[str, int]:
    m = {}
    with open(CLASS_IND, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                m[parts[1]] = int(parts[0]) - 1
    return m

def _read_split(split_path: Path, class_map: Dict[str,int]) -> List[VideoItem]:
    items: List[VideoItem] = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            rel = line.strip().split()[0]
            cls = rel.split("/")[0]
            items.append(VideoItem(str(UCF_VIDEOS / rel), class_map[cls]))
    return items

def _sample_indices(n_frames: int, T: int, mode: str) -> List[int]:
    if n_frames <= 0:
        return []
    if n_frames < T:
        idx = list(np.linspace(0, max(0, n_frames-1), T).astype(int))
        return idx
    base = np.linspace(0, n_frames - 1, T).astype(int)
    if mode == "train":

        jitter = np.random.randint(-2, 3, size=T)
        base = np.clip(base + jitter, 0, n_frames - 1)
    return base.tolist()

def _read_clip(path: str, idxs: List[int]) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            cap.release()
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
        frames.append(rgb.astype(np.float32) / 255.0)
    cap.release()
    return np.stack(frames, axis=0)  

def _gen(items: List[VideoItem], T: int, mode: str):
    for it in items:

        cap = cv2.VideoCapture(it.path)
        if not cap.isOpened():
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total <= 0:
            continue
        idxs = _sample_indices(total, T, mode)
        clip = _read_clip(it.path, idxs)
        if clip is None:
            continue
        yield clip, np.int32(it.label)

def make_sequence_datasets(
    T: int = 16,
    batch_size: int = 8,
    shuffle_buffer: int = 1024,
):
    class_map = _read_class_map()
    n_classes = len(class_map)
    train_items = _read_split(TRAIN_SPLIT, class_map)
    val_items   = _read_split(TEST_SPLIT,  class_map)

    sig = (
        tf.TensorSpec(shape=(T, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    train_ds = tf.data.Dataset.from_generator(
        lambda: _gen(train_items, T, "train"),
        output_signature=sig
    ).shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: _gen(val_items, T, "val"),
        output_signature=sig
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, n_classes
