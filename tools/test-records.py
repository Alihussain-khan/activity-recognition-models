import tensorflow_datasets as tfds
b = tfds.builder("ucf101", data_dir="~/bhome/group8/tensorflow_datasets")
print({k:v.num_examples for k,v in b.info.splits.items()})  # expect {'train': 9537, 'test': 3783}
