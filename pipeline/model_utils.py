import keras_tuner as kt
import tensorflow as tf
from tfx.components.trainer.fn_args_utils import FnArgs 
from configs.config import PIPELINE_NAME, TARGET, FEATURES
from kerastuner.engine import base_tuner 
from typing import NamedTuple, Dict, Text, Any 
import tensorflow_transform as tft

N_EPOCHS = 20
BATCH_SIZE = 40

def transformed_name(key):
    key = key.replace('-', '_')
    return key + '_xf'

def _gzip_reader_fn(filenames):
    
    return tf.data.TFRecordDataset(filenames, compression_type = 'GZIP')

def _input_fn(file_pattern, 
              tf_transform_output,
              num_epochs=N_EPOCHS,
              batch_size=BATCH_SIZE) -> tf.data.Dataset:
  
  
  transformed_feature_spec =(
          tf_transform_output.transformed_feature_spec().copy()
      )    
  
  dataset  = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader = _gzip_reader_fn,
      num_epochs=num_epochs,
      label_key=transformed_name(TARGET)
  )

  return dataset

def model_builder(hp):

    num_hidden_layers = hp.Int('hidden_layers', min_value=1, max_value=5)

    input_numeric = [
        tf.keras.layers.Input(name='numeric_' + str(i), shape=(1,), dtype=tf.float32) for i in range(FEATURES)
    ]

    deep = tf.keras.layers.concatenate(input_numeric)

    for i in range(num_hidden_layers):
        num_nodes = hp.Int('unit_'+str(i), min_value=8, max_value=256, step=64)
        deep = tf.keras.layers.Dense(num_nodes, activation='relu')(deep)

    # Output layer
    output = tf.keras.layers.Dense(1, activation='sigmoid')(deep)

    model = tf.keras.models.Model(inputs=input_numeric, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-2, max_value=1e-1, sampling='LOG')),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def tuner_fn(fn_args: FnArgs):
    tuner  = kt.Hyperband(
        model_builder,
        objective='val_binary_accuracy',
        max_epochs=N_EPOCHS,
        factor =2,
        directory = fn_args.working_dir,
        project_name = 'kt_hyperband'
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                            ('fit_kwargs', Dict[Text,Any])])

    # load transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = _input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = _input_fn(fn_args.eval_files, tf_transform_output, 10)


    return TunerFnResult(
        tuner = tuner,
        fit_kwargs={
            "callbacks":[stop_early],
            "x": train_set,
            "validation_data": val_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps
        }
    )