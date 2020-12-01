DATA_DIR=path/to/data
MODEL_DIR=path/to/model
CODE_DIR=path/to/research/object_detection

TRAIN_IMAGE_PATH=$DATA_DIR/my_data/images/train
TEST_IMAGE_PATH=$DATA_DIR/my_data/images/test

ANNOTATION_PATH=$DATA_DIR/my_data/annotations
LABELMAP_PATH=$ANNOTATION_PATH/label_map.pbtxt
CLASS_NAME='myLabel'
TRAIN_TFRECORD_PATH=$ANNOTATION_PATH/train.record
TEST_TFRECORD_PATH=$ANNOTATION_PATH/test.record

MODEL_PATH=$MODEL_DIR/my_model
PIPELINE_PATH=$MODEL_PATH/pipeline.config

SOURCE_MODEL_PATH=$MODEL_DIR/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
SOURCE_PIPELINE_PATH=$SOURCE_MODEL_PATH/pipeline.config

EXPORTED_MODEL_PATH=$MODEL_DIR/my_exported_model

TRAIN_PY=$CODE_DIR/model_main_tf2.py
EXPORT_PY=$CODE_DIR/exporter_main_v2.py

python demo_config.py $LABELMAP_PATH $CLASS_NAME $PIPELINE_PATH $SOURCE_MODEL_PATH $SOURCE_PIPELINE_PATH $TRAIN_TFRECORD_PATH $TEST_TFRECORD_PATH
python generate_tfrecord.py -x $TRAIN_IMAGE_PATH -l $LABELMAP_PATH -o $TRAIN_TFRECORD_PATH
python generate_tfrecord.py -x $TEST_IMAGE_PATH -l $LABELMAP_PATH -o $TEST_TFRECORD_PATH

python $TRAIN_PY --model_dir=$MODEL_PATH --pipeline_config_path=$MODEL_PATH/pipeline.config

python $EXPORT_PY --input_type=image_tensor --pipeline_config_path=$MODEL_PATH/pipeline.config --trained_checkpoint_dir $MODEL_PATH --output_directory $EXPORTED_MODEL_PATH
python demo_detect.py $EXPORTED_MODEL_PATH