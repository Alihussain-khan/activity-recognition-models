from __future__ import annotations
from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds

ROOT = Path.home() / "bhome" / "group8"
TFDS_DIR = ROOT / "tensorflow_datasets"
IMG_H, IMG_W = 224, 224

def _resize_norm(img: tf.Tensor) -> tf.Tensor:

    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR)
    return tf.cast(img, tf.float32) / 255.0


def make_datasets_single_tfds(batch_size: int = 32, shuffle_buffer: int = 4096):
    import tensorflow_datasets as tfds
    builder_dir = str(TFDS_DIR)

    train = tfds.load("ucf101", split="train", data_dir=builder_dir, shuffle_files=True)
    test  = tfds.load("ucf101", split="test",  data_dir=builder_dir, shuffle_files=False)

    def _resize_norm(img: tf.Tensor) -> tf.Tensor:
        img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR)
        return tf.cast(img, tf.float32) / 255.0

    def _pick_one_train(sample):
        vid = sample["video"]             
        lbl = tf.cast(sample["label"], tf.int32)
        t = tf.shape(vid)[0]
        idx = tf.random.uniform((), 0, tf.maximum(t, 1), dtype=tf.int32)
        frame = _resize_norm(vid[idx])
        return frame, lbl

    def _pick_one_eval(sample):
        vid = sample["video"]; lbl = tf.cast(sample["label"], tf.int32)
        t = tf.shape(vid)[0]
        idx = tf.maximum(0, (t // 2) - 1)
        frame = _resize_norm(vid[idx])
        return frame, lbl

    train_ds = (train
        .shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        .map(_pick_one_train, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE))

    val_ds = (test
        .map(_pick_one_eval, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds, 101



def make_sequence_datasets_tfds(T: int = 16, batch_size: int = 8, shuffle_buffer: int = 2048):
    import tensorflow_datasets as tfds
    builder_dir = str(TFDS_DIR)

    train = tfds.load("ucf101", split="train", data_dir=builder_dir, shuffle_files=True)
    test  = tfds.load("ucf101", split="test",  data_dir=builder_dir, shuffle_files=False)

    def _resize_norm(img: tf.Tensor) -> tf.Tensor:
        img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR)
        return tf.cast(img, tf.float32) / 255.0

    def _sample_indices(t, T, train_mode: bool):
        base = tf.cast(tf.linspace(0.0, tf.maximum(tf.cast(t, tf.float32) - 1.0, 0.0), T), tf.int32)
        if train_mode:
            jitter = tf.random.uniform([T], minval=-2, maxval=3, dtype=tf.int32)
            base = tf.clip_by_value(base + jitter, 0, tf.maximum(t - 1, 0))
        return base

    def _to_clip(sample, train_mode: bool):
        vid = sample["video"]; lbl = tf.cast(sample["label"], tf.int32)
        t = tf.shape(vid)[0]
        idx = _sample_indices(t, T, train_mode)
        clip = tf.gather(vid, idx) 
        clip = tf.map_fn(_resize_norm, clip, fn_output_signature=tf.float32)
        return clip, lbl

    train_ds = (train
        .shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        .map(lambda s: _to_clip(s, True), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))

    val_ds = (test
        .map(lambda s: _to_clip(s, False), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds, 101
