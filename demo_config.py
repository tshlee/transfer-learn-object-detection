import os
import sys

from google.protobuf import text_format
from object_detection.protos.pipeline_pb2 import TrainEvalPipelineConfig 
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem

LABELMAP_PATH, CLASS_NAME, PIPELINE_PATH, SOURCE_MODEL_PATH, SOURCE_PIPELINE_PATH, TRAIN_TFRECORD_PATH, TEST_TFRECORD_PATH = sys.argv[1:8]

#Save labelmap
labelmap = StringIntLabelMap()
labelmap.item.append(StringIntLabelMapItem(id=1, name=CLASS_NAME))
with open(LABELMAP_PATH, 'w') as f:
    f.write(text_format.MessageToString(labelmap))

#Edit source pipeline and save as pipeline
pipeline = TrainEvalPipelineConfig()
with open(SOURCE_PIPELINE_PATH, 'r') as f:
    text_format.Merge(f.read(), pipeline)
pipeline.model.ssd.num_classes = 1
pipeline.train_input_reader.label_map_path = LABELMAP_PATH
pipeline.train_input_reader.tf_record_input_reader.input_path[0] = TRAIN_TFRECORD_PATH
pipeline.train_config.fine_tune_checkpoint = os.path.join(SOURCE_MODEL_PATH, 'checkpoint/ckpt-0')
pipeline.train_config.fine_tune_checkpoint_type = 'detection'
pipeline.train_config.batch_size = 4
pipeline.train_config.use_bfloat16 = False
pipeline.eval_input_reader[0].label_map_path = LABELMAP_PATH
pipeline.eval_input_reader[0].tf_record_input_reader.input_path[0] = TEST_TFRECORD_PATH
pipeline.eval_config.metrics_set[0] = 'coco_detection_metrics'
pipeline.eval_config.use_moving_averages = False
with open(PIPELINE_PATH, 'w') as f:
    f.write(text_format.MessageToString(pipeline))