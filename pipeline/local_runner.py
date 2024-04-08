from configs.config import DATA_PATH, PIPELINE_ROOT_PATH, PIPELINE_NAME, TRANSFORM_MODULE_PATH, SERVING_MODEL_DIR, METADATA_PATH, TRAIN_MODULE_PATH
from absl import logging
from tfx import v1 as tfx
from data_pipeline import create_pipeline


def run():
  """Define a local pipeline."""

  tfx.orchestration.LocalDagRunner().run(
          create_pipeline(
          pipeline_name=PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT_PATH,
          data_path=DATA_PATH,
          serving_dir=SERVING_MODEL_DIR
          )
        )


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()