# Teeth detector

![Image 1](public/images/1.png "Image 1")
![Image 2](public/images/7.png "Image 2")
![Image 3](public/images/4.png "Image 3")

Dataset is private for the moment, but was made with a stomatologist surgeon, using VoTT for labeling. The export was made under the `Tensorflow Pascal VOC` format

The project is divided into two parts:
1. Extract the labels from the teeth and train a CNN to identify the tooth number (`train.py`). This was a quick experiment and no effort is currently spent on it
2. Train an object detection pipeline with Tensorflow in order to detect tooth restoration, endodotic treatment and implants


## Installation

- Download the datasets from the google drive
- Git clone https://github.com/tensorflow/models/, and run ```export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim``` in   `/models/research`


## TODO
- data preprocessing
- configure Object Detection Pipeline
- run a simple model on the data
- run a better model
- automatise installation
- augment dataset, especially for implants


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
