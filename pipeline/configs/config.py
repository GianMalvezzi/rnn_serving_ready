import os

PIPELINE_ROOT_PATH = os.path.abspath('pipeline/root/')
TRANSFORM_MODULE_PATH = os.path.abspath('pipeline/transform_utils.py')
DATA_PATH = os.path.abspath('data/train.csv')
TRAIN_MODULE_PATH = os.path.abspath('pipeline/train_utils.py')

PIPELINE_NAME = 'RNN'

METADATA_PATH = os.path.join(PIPELINE_ROOT_PATH, 'tfx_metadata', PIPELINE_NAME,
                             'metadata.db')

SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT_PATH, 'serving_model')

TARGET = 'Outcome'
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "Diabetes",
    "PedigreeFunction",
    "Age"
]