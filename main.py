import random
import time
import typing

import click
import numpy as np

import convert
from convert import split
import metrics


@click.group()
def rex():
    pass


@rex.command('convert')
@click.option('-f', '--file', 'input_paths', multiple=True, required=True)
@click.option('-o', '--output', 'output_path', required=True)
@click.option('--in-format', 'input_format',
              required=True,
              type=click.Choice(convert.available_importers, case_sensitive=False))
@click.option('--out-format', 'output_format', default='line-json',
              required=False, type=click.Choice(convert.available_exporters, case_sensitive=False))
@click.option('--top-n', 'top_n', required=False, type=click.IntRange(min=0))
@click.option('--skip', 'skip', required=False, type=click.IntRange(min=0))
@click.option('--split', 'splits', multiple=True, type=split.SplitParam(),
              help='Define a split of this dataset. If set, the output will '
                   'be interpreted as directory. Each split has to be in the form '
                   '<split_name>:<split_ratio>, where split_name is the output filename '
                   'and split_ratio is a value in [0;1]. Summing all split_ratios has '
                   'to result in 1.0. You can use the asterisk (*) to assign the remaining '
                   'samples to a split. '
                   '--split "train.json:.8" --split "valid.json:.2" will result in a file'
                   'train.json containing 80% of the dataset, and a file test.json containing '
                   '20%. --split "train.json:.8" --split "valid.json:*" will result in the same '
                   'split. --split "train.json:*" --split "valid.json:*" would result in two '
                   'files with 50% of the dataset each.')
@click.option('--stratified', is_flag=True)
@click.option('--shuffle', is_flag=True)
@click.option('--seed', type=int,
              help='Seed for random generators, to get reproducible results, '
                   'defaults to current seconds since epoch.')
def rex_convert(input_paths: typing.List[str], output_path: str,
                input_format: str, output_format: str,
                top_n: int = None, skip: int = None,
                splits: typing.List[typing.Union[split.Split, split.AutoSplit]] = None,
                shuffle: bool = False,
                stratified: bool = False,
                seed: int = None):
    if seed is None:
        seed = int(time.time())
    np.random.seed(seed)
    random.seed(seed)

    if skip is None:
        skip = 0

    resolved_splits = split.resolve_splits(splits)
    importer = convert.available_importers[input_format](top_n=top_n, skip=skip)
    exporter = convert.available_exporters[output_format]()
    convert.main(input_paths, output_path, importer, exporter, resolved_splits, shuffle, stratified)


@rex.command('evaluate')
@click.option('-f', '--file', 'input_paths', multiple=True, required=True)
@click.option('--penalize', 'penalize', is_flag=True)
@click.option('--original-data', 'original_data', required=False, type=str, multiple=True)
@click.option('--original-data-format', 'original_data_format', required=False,
              type=click.Choice(convert.available_exporters, case_sensitive=False))
@click.option('--format', 'result_format', required=True,
              type=click.Choice(metrics.available_result_parsers, case_sensitive=False))
@click.option('--f1-type', 'f1_type', type=click.Choice(['micro-global', 'micro-local', 'macro-global']))
def rex_evaluate(input_paths: typing.List[str], result_format: str, f1_type: str,
                 penalize: bool = False, original_data: typing.List[str] = None, original_data_format: str = None):
    file_parser = metrics.available_result_parsers[result_format]
    parsed = list(file_parser.parse(input_paths))
    if penalize:
        assert original_data is not None
        assert original_data_format is not None
        original_dataset = convert.available_importers[original_data_format]().load(original_data)
        metrics.BaseParser.insert_missing_samples(parsed, original_dataset)

    click.echo(metrics.calculate_f1(parsed, metrics.ExactRelationMatcher(), f1_type))


if __name__ == '__main__':
    rex()
