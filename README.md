# Teeth detector

## Installation

- Download the noor dataset from the google drive, and save it as `noor_dataset`
- Copy the `object_detection` folder from https://github.com/tensorflow/models


## TODO
- data preprocessing
- configure Object Detection Pipeline
- run a simple model on the data
- run a better model
- automatise installation


# Commands

```
python ../models/research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/Users/clementjoudet/Desktop/dev/teeth/models/model/ssd.config \
    --train_dir="/Users/clementjoudet/Desktop/dev/teeth/models/model/train"
```

```
python ../models/research/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=/Users/clementjoudet/Desktop/dev/teeth/models/model/ssd.config \
    --checkpoint_dir="/Users/clementjoudet/Desktop/dev/teeth/models/model/train/" \
    --eval_dir="/Users/clementjoudet/Desktop/dev/teeth/models/model/eval"
```
