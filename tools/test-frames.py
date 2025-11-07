import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="4"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import tensorflow_datasets as tfds
import numpy as np

ds = tfds.load("ucf101", split="train", data_dir="~/bhome/group8/tensorflow_datasets")
lengths = []
for i, ex in enumerate(ds.take(20)):    
    lengths.append(int(ex["video"].shape[0]))
print("Sampled frame counts:", lengths)
print("min/max frames in sample:", np.min(lengths), np.max(lengths))
