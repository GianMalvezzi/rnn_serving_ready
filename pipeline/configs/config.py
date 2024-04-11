import os

PIPELINE_ROOT_PATH = os.path.abspath('pipeline/root/')
TRANSFORM_MODULE_PATH = os.path.abspath('pipeline/transform_utils.py')
DATA_PATH_TRAIN = os.path.abspath('data/train.csv')
DATA_PATH_EVAL = os.path.abspath('data/eval.csv')
DATA_PATH_TEST = os.path.abspath('data/test.csv')
TRAIN_MODULE_PATH = os.path.abspath('pipeline/train_utils.py')
LOCAL_RUNNER_PATH = os.path.abspath('pipeline/local_runner.py')


PIPELINE_NAME = 'RNN'

METADATA_PATH_TRAIN = os.path.join(PIPELINE_ROOT_PATH, 'tfx_metadata', PIPELINE_NAME,
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