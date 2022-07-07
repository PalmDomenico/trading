import numpy as np
import os

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf

assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging

logging.set_verbosity(logging.ERROR)
spec = model_spec.get('efficientdet_lite0')
train_data = object_detector.DataLoader.from_pascal_voc(
    'test/train',
    'test/train',
    ['denari', 'coppe']
)

val_data = object_detector.DataLoader.from_pascal_voc(
    'test/validate',
    'test/validate',
    ['denari', 'coppe']
)
model = object_detector.create(train_data, model_spec=spec, batch_size=3, train_whole_model=True, epochs=20, validation_data=val_data)
model.export(export_dir='.', tflite_filename='android.tflite')