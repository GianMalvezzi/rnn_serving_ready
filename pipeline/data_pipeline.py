import tfx
from tfx.v1.dsl import Pipeline, Resolver, Channel
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Evaluator, Pusher, Trainer, Tuner
import tensorflow_model_analysis as tfma
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel as TFXChannel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.experimental import latest_blessed_model_resolver
from configs.config import TRANSFORM_MODULE_PATH, DATA_PATH


def create_pipeline(
        pipeline_name,
        pipeline_root,
        data_path,
        serving_dir,
        beam_pipeline_args = None,
        metadata_connection_config=None
)-> tfx.v1.dsl.Pipeline:
    
    """Create a TFX logical pipeline.   
    Args:
        pipeline_name: name of the pipeline
        pipeline_root: directory to store pipeline artifacts
        data_root: input data directory
        module_file: module file to inject customized logic into TFX components
        metadata_path: path to local sqlite database file
        beam_pipeline_args: arguments for Beam powered components
    """
    
    components =[]

    example_gen = tfx.components.CsvExampleGen(input_base = data_path,
                                               input_config=example_gen_pb2.Input(splits=[
                                example_gen_pb2.Input.Split(name='train', pattern='train.csv'),
                                example_gen_pb2.Input.Split(name='eval', pattern='eval.csv')
                            ]))
    
    components.append(example_gen)
    
    # Computes statistics over data for visualization and example validation.
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples'])   
    
    components.append(statistics_gen)
    
    # Generates schema based on statistics files.
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True)   
    
    components.append(schema_gen)

    # Performs anomaly detection based on statistics and data schema.
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'],
    )

    components.append(example_validator)

    transform = Transform(
        examples = example_gen.outputs['examples'],
        schema = schema_gen.outputs['schema'],
        module_file=TRANSFORM_MODULE_PATH
    )
    
    components.append(transform)

    trainer = Trainer(
        module_file=DATA_PATH,
        transformed_examples=transform.outputs['transformed_examples'],
        schema=transform.outputs['transform_output'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=1000),
        eval_args=trainer_pb2.EvalArgs(num_steps=500))

    components.append(trainer)

    tuner = Tuner(
        module_file = DATA_PATH,
        examples = transform.outputs['transformed_examples'],
        transform_graph = transform.outputs['transform_graph'],
        schema = schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=500), 
        eval_args=trainer_pb2.EvalArgs(num_steps=100)
        )

    components.append(tuner)

    model_resolver = Resolver(
        strategy_class = latest_blessed_model_resolver.LatestBlessedModelStrategy,
        model = Channel(type=Model),
        model_blessing = Channel(
            type = ModelBlessing)
            )
            
    components.append(model_resolver)

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='sentiment')],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
                tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(
                                class_name='BinaryAccuracy',
                                threshold=tfma.MetricThreshold(
                                    value_threshold=tfma.GenericValueThreshold(
                                        lower_bound={'value': 0.6}),
                                    # Change threshold will be ignored if there is no
                                    # baseline model resolved from MLMD (first run).
                                    change_threshold=tfma.GenericChangeThreshold(
                                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                        absolute={'value': -1e-10})))
                        ])])
   
    evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config)

    components.append(evaluator)

    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_dir)))
        
    components.append(pusher)

    return Pipeline(

        pipeline_name = pipeline_name,
        pipeline_root = pipeline_root,
        components=components,
        beam_pipeline_args=beam_pipeline_args,
        metadata_connection_config=metadata_connection_config
    )
