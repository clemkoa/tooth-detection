# Commands

```
python ../models/research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=/Users/clementjoudet/Desktop/dev/tooth-detection/models/cloud/ssd-local.config \
    --train_dir="/Users/clementjoudet/Desktop/dev/tooth-detection/models/cloud/train"
```

```
python ../models/research/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=/Users/clementjoudet/Desktop/dev/tooth-detection/models/cloud/ssd-local.config \
    --checkpoint_dir="/Users/clementjoudet/Desktop/dev/tooth-detection/models/cloud/train/" \
    --eval_dir="/Users/clementjoudet/Desktop/dev/tooth-detection/models/cloud/eval"
```


# Gcloud

Train job
```
gcloud ml-engine jobs submit training tooth_jood_`date +%s` \
    --runtime-version 1.5 \
    --job-dir=gs://tooth-jood/train/ \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --config /Users/clementjoudet/Desktop/dev/tooth-detection/models/cloud/cloud.yml \
    --region europe-west1 \
    -- \
    --train_dir=gs://tooth-jood/train/ \
    --pipeline_config_path=gs://tooth-jood/faster_rcnn_resnet50_coco.config
```

Stream logs

```
gcloud ml-engine jobs stream-logs JOB_ID
```

Eval job
```
gcloud ml-engine jobs submit training object_detection_eval_`date +%s` \
    --runtime-version 1.5 \
    --job-dir=gs://tooth-jood/data/ \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.eval \
    --region us-central1 \
    --scale-tier BASIC_GPU \
    -- \
    --checkpoint_dir=gs://tooth-jood/data/ \
    --eval_dir=gs://tooth-jood/data/eval/ \
    --pipeline_config_path=gs://tooth-jood/data/ssd.config
```

Cancel job

```
gcloud ml-engine jobs cancel JOB_ID
```

See job list

```
gcloud ml-engine jobs list
```
