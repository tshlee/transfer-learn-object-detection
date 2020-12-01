import os
import sys
import contextlib

import PIL
import numpy as np
import tensorflow as tf

@contextlib.contextmanager
def process_context(saved_model_path):
    detect_fn = tf.saved_model.load(saved_model_path)
    def process_fn(image_path):
        image_np = np.array(PIL.Image.open(image_path))
        image_tensor = tf.convert_to_tensor(image_np)
        #as singleton:
        image_tensor = image_tensor[tf.newaxis, ...]
        detections = detect_fn(image_tensor)
        #from singleton:
        M = int(detections.pop('num_detections'))
        detections = {k: v[0,:M].numpy() for k,v in detections.items()}
        detections['num_detections'] = M
        return detections
    yield process_fn

def image_path_source():
    yield '/path/to/image1.jpg'
    yield '/path/to/image2.jpg'

EXPORTED_MODEL_PATH = sys.argv[1]
with process_context(os.path.join(EXPORTED_MODEL_PATH, 'saved_model')) as process_fn:
    for image_path in image_path_source():
        detections = process_fn(image_path)