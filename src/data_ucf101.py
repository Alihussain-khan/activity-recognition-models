
from __future__ import annotations
import os, cv2, random, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

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

    class_map: Dict[str, int] = {}
    with open(CLASS_IND, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                idx_str, name = parts
                class_map[name] = int(idx_str) - 1
    return class_map

def _read_split_file(split_path: Path, class_map: Dict[str, int], is_train: bool) -> List[VideoItem]:

    items: List[VideoItem] = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            rel = parts[0]
            cls_name = rel.split("/")[0]
            label = class_map[cls_name]
            full = str(UCF_VIDEOS / rel)
            items.append(VideoItem(path=full, label=label))
    return items

def _center_frame_index(total_frames: int) -> int:
    return max(0, total_frames // 2)

def _read_one_frame(video_path: str, mode: str = "train") -> np.ndarray | None:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    if mode == "train":
        idx = random.randint(0, max(0, total - 1))
    else:
        idx = _center_frame_index(total)

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        return None

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    frame_rgb = frame_rgb.astype(np.float32) / 255.0
    return frame_rgb

def _generator(items: List[VideoItem], mode: str) -> Iterable[Tuple[np.ndarray, np.int32]]:

    for it in items:
        img = None
        for _ in range(3):
            img = _read_one_frame(it.path, mode=mode)
            if img is not None:
                break
        if img is None:
            continue
        yield img, np.int32(it.label)

def make_datasets(
    batch_size: int = 32,
    shuffle_buffer: int = 2048,
    num_parallel_calls: int | None = None
):

    class_map = _read_class_map()
    num_classes = len(class_map)

    train_items = _read_split_file(TRAIN_SPLIT, class_map, is_train=True)
    val_items   = _read_split_file(TEST_SPLIT,  class_map, is_train=False)

    if num_parallel_calls is None:
        num_parallel_calls = tf.data.AUTOTUNE

    output_signature = (
        tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    train_ds = tf.data.Dataset.from_generator(
        lambda: _generator(train_items, mode="train"),
        output_signature=output_signature,
    )
    val_ds = tf.data.Dataset.from_generator(
        lambda: _generator(val_items, mode="val"),
        output_signature=output_signature,
    )

    train_ds = (
        train_ds
        .shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, num_classes
