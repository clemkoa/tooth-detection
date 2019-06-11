# Commands

```
python ../models/research/object_detection/model_main.py \
    --pipeline_config_path=/Users/clementjoudet/Desktop/dev/tooth-detection/models/transfer/faster_rcnn_resnet50_coco.config \
    --model_dir=/Users/clementjoudet/Desktop/dev/tooth-detection/models/transfer/new_version \
    --num_train_steps=100000 \
    --alsologtostderr
```

# Gcloud

Train job
```
gcloud ml-engine jobs submit training tooth_jood_`date +%Y-%m-%d:%H-%M-%S` \
    --runtime-version 1.9 \
    --job-dir=gs://tooth-jood/new/ \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_main \
    --region europe-west1 \
    --config /Users/clementjoudet/Desktop/dev/tooth-detection/models/cloud/cloud.yml \
    -- \
    --model_dir=gs://tooth-jood/new/ \
    --pipeline_config_path=gs://tooth-jood/index.config
```

Export model for inference
```
# From tensorflow/models/research/
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/Users/clementjoudet/Desktop/dev/tooth-detection/models/index/index_local.config
TRAINED_CKPT_PREFIX=/Users/clementjoudet/Desktop/dev/tooth-detection/models/index/cloud/new/model.ckpt-13842
EXPORT_DIR=/Users/clementjoudet/Desktop/dev/tooth-detection/models/index/cloud/inference
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
```

Stream logs

```
gcloud ml-engine jobs stream-logs JOB_ID
```

Cancel job

```
gcloud ml-engine jobs cancel JOB_ID
```

See job list

```
gcloud ml-engine jobs list
```
