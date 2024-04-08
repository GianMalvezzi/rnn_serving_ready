from configs.config import DATA_PATH, PIPELINE_ROOT_PATH, PIPELINE_NAME, TRANSFORM_MODULE_PATH, SERVING_MODEL_DIR, METADATA_PATH, TRAIN_MODULE_PATH
from absl import logging
from tfx import v1 as tfx
from pipeline.data_pipeline import run_pipeline


def run():
  """Define a local pipeline."""

  tfx.orchestration.LocalDagRunner().run(
      run_pipeline.create_pipeline(
          pipeline_name=PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT_PATH,
          data_path=DATA_PATH,
          preprocessing_module=TRANSFORM_MODULE_PATH,
          tuner_path=TRAIN_MODULE_PATH,
          training_module=TRAIN_MODULE_PATH,
          serving_model_dir=SERVING_MODEL_DIR,
          metadata_connection_config=tfx.orchestration.metadata
          .sqlite_metadata_connection_config(METADATA_PATH)
          )
        )


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run()