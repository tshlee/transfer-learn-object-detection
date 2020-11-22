DATA_DIR=/path/to/data
MODEL_DIR=/path/to/model
#CODE_DIR=.
CODE_DIR=/path/to/research/object_detection

TRAIN_IMAGE_PATH=$DATA_DIR/my_data/images/train  #my-demo/images/train #incl *.xml
TEST_IMAGE_PATH=$DATA_DIR/my_data/images/test
ANNOTATION_PATH=$DATA_DIR/my_data/annotations  #*.csv *.record

MODEL_PATH=$MODEL_DIR/my_model  #my-demo/models/my_model
SOURCE_MODEL_PATH=$MODEL_DIR/pretrained_model  #my-demo/pre-trained-models
EXPORTED_MODEL_PATH=$MODEL_DIR/my_exported_model  #my-demo/exported-models

TRAIN_PY=$CODE_DIR/model_main_tf2.py
EXPORT_PY=$CODE_DIR/exporter_main_v2.py

python $TRAIN_PY --model_dir=$MODEL_PATH --pipeline_config_path=$MODEL_PATH/pipeline.config
python $EXPORT_PY --input_type=image_tensor --pipeline_config_path=$MODEL_PATH/pipeline.config --trained_checkpoint_dir $MODEL_PATH --output_directory $EXPORTED_MODEL_PATH
python demo_detect.py $EXPORTED_MODEL_PATH