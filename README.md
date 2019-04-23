# Teeth detector

![Image 1](public/images/1.png "Image 1")
![Image 2](public/images/2.png "Image 2")
![Image 3](public/images/3.png "Image 3")

Dataset is private for the moment, but was made with a stomatologist surgeon, using VoTT for labeling. The export was made under the `Tensorflow Pascal VOC` format

The project is divided into two parts:
1. Extract the labels from the teeth and train a CNN to identify the tooth number (`train.py`). This was a quick experiment and no effort is currently spent on it
2. Train an object detection pipeline with Tensorflow in order to detect tooth restoration, endodotic treatment and implants


## Installation

- Download the datasets from the google drive (datasets are private at the moment)
- Install tensorflow object detection: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
- Install Cloud SDK to run on google cloud https://cloud.google.com/sdk/


```
pip install -r requirements.txt

# Tensorflow Object Detection API
git clone git@github.com:tensorflow/models.git

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <path_to_tensorflow>/models/research/

# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```


## Training

```
python <path_to_tensorflow>/models/research/object_detection/model_main.py \
    --pipeline_config_path=<path_to_tooth-detection>/tooth-detection/models/transfer/faster_rcnn_resnet50_coco.config \
    --model_dir=<path_to_tooth-detection>/tooth-detection/models/transfer/new_version \
    --num_train_steps=100000 \
    --alsologtostderr
```

## Inference

```
python inference.py
```
