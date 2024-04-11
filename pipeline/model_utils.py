import keras_tuner as kt
import tensorflow as tf
from tfx.components.trainer.fn_args_utils import FnArgs 
from configs.config import PIPELINE_NAME, TARGET, FEATURES
from kerastuner.engine import base_tuner 
from typing import NamedTuple, Dict, Text, Any 
import tensorflow_transform as tft
import os
from tfx.components.trainer.executor import TrainerFnArgs

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

def get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(TARGET)  

        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)

        
        return model(transformed_features)

    return serve_tf_examples_fn


def raw_input_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def raw_input(features: Dict[Text, tf.Tensor]):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(TARGET)

        output_features = {}

        for key, spec in feature_spec.items():
            if isinstance(spec, tf.io.VarLenFeature):
                output_features[key] = tf.sparse.from_dense(features[key])
            else:
                output_features[key] = features[key]

        transformed_features = model.tft_layer(output_features)

        outputs = model(transformed_features)
        return {
            'probabilities': outputs,
            'label_key': tf.argmax(outputs, axis=1),
            'prediction_confidence': tf.reduce_max(outputs, axis=1)
        }

    return raw_input

def run_fn(fn_args: TrainerFnArgs):
    """Train the model based on given args.

    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Load datasets
    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 40)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 40)

    # Retrieve hyperparameters
    hparams = tuner_fn(fn_args).fit_kwargs.get('tuner').get_config()['hyperparameters']

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = model_builder(hparams)

    try:
        log_dir = fn_args.model_run_dir
    except KeyError:
        log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

    # Write logs to path
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch')

    # Train the model
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback])

    # Define serving signatures
    serving_default_signature = _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
    signatures = {'serving_default': serving_default_signature}

    # Save the model
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)


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

    train_set = _input_fn(fn_args.train_files, tf_transform_output)
    val_set = _input_fn(fn_args.eval_files, tf_transform_output)


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