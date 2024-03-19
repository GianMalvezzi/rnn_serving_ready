from pipeline.configs.loader import load_config
from pipeline.data_pipeline import create_pipeline

config = load_config()
pipeline = create_pipeline(config['PATHS']['TRANSFORM_MODULE_PATH'])
print(pipeline)