import tfx
from tfx.v1.dsl import Pipeline
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform
from pipeline.configs.loader import load_config

def create_pipeline(data_path):
    config = load_config()

    example_gen = CsvExampleGen(input_base=data_path)
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

    example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema'])

    transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=config['PATHS']['TRANSFORM_MODULE_PATH'])

    return Pipeline(
        pipeline_name='light_gbm',
        pipeline_root='pipeline/root',
        components=[example_gen, statistics_gen, schema_gen, transform]
    )