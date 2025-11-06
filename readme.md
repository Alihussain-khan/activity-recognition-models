Hi, here I will try to take you through what lies where. :)

One thing, this repository is not exact copy of the repo on the remote server. It is missing the models, logs, checkpoints and dataset. 
Some of the failed experiements stayed on the remoter server :)

Some of the scripts only need to be ran once, and some you need to get the environment running.



## directory structure
- "evaluation-scripts" are used to evaluate each model
- "figure-scripts" are used to make figures (eg python make_figs_from_dump_cleansplit.py lstm)
- "figures" have the figures
- "metrics" are test/eval metrics
- "models" have the models
- "src" data loader
- "tools" have summary and tensors to csv results


# getting started (how to run)
#### run it from group8
check the gpu status using
- nvidia-smi

activate the enviroment / gpu drivers
- uenv miniconda3-py39
- conda activate actrec314
- uenv cuda-11.8.0
- uenv list



## load dataset (usefull but i downloaded manually)
  
python - <<'PY'
import tensorflow_datasets as tfds
builder = tfds.builder("ucf101")
builder.download_and_prepare()
print("Dataset ready:", builder.info.splits.keys())
PY

## build records 

python - <<'PY'
import os, tensorflow_datasets as tfds
from tensorflow_datasets.core.download import DownloadConfig

manual = os.path.expanduser('~/bhome/group8/tensorflow_datasets/downloads/manual')
data   = os.path.expanduser('~/bhome/group8/tensorflow_datasets')

print("Manual dir:", manual)
print("Data dir  :", data)
print("Manual contents:", sorted(os.listdir(manual))[:6], "...")

builder = tfds.builder("ucf101", data_
dir=data)
builder.download_and_prepare(download_config=DownloadConfig(
    manual_dir=manual, 
    verify_ssl=False     
))
print("Built at:", builder.data_dir)
print("Splits:", list(builder.info.splits.keys()))
PY


## the models have tags that can change the parameters from the terminal rather than changing the code, these parameters where used.


#### CNN:
sf python ~/bhome/group8/train_single_frame_inception_v1.py   --batch 32   --epochs 40   --lr 1e-3s

#### CNN+LSTM:
python ~/bhome/group8/train_cnn_lstm_tuned.py --frames 16 --batch 8 --epochs 20 --lr 3e-4 --label_smoothing 0.05 --fine_tune_after 0 --lstm_units 512 --dropout_td 0.3 --dropout_head 0.3 --weight_decay 3e-4 --clip_norm 1.0

#### i3d:
 python ~/bhome/group8/train_i3d_pretrained.py \
  --frames 32 \
  --batch 6 \
  --epochs 30 \
  --base_lr 3e-4 \
  --mixed_precision

