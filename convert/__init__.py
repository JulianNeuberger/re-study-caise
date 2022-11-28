import os
import random
import typing

import click

import model
from convert import importer, exporter, sanity, split

available_importers: typing.Dict[str, typing.Type[importer.BaseImporter]] = {
    'nyt10': importer.Nyt10Importer,
    'semeval': importer.SemEvalImporter,
    'conll': importer.ConLL04Importer,
    'tacred': importer.TacredImporter,
    'fewrel': importer.FewRelImporter,
    'line-json': importer.JsonLinesImporter
}
available_exporters: typing.Dict[str, typing.Type[exporter.BaseExporter]] = {
    'line-json': exporter.LineJsonExporter,
    'pfn-custom': exporter.LineJsonExporter,
    'pfn': exporter.LineJsonExporter,
    'two-are-better-than-one': exporter.TwoAreBetterThanOneExporter,
    'joint-er': exporter.JointERExporter,
    'rsan': exporter.RSANExporter,
    'brat': exporter.BratExporter,
    'span-rel': exporter.BratExporter,
    'progressive-multitask': exporter.ProgressiveMultiTaskExporter,
    'casrel': exporter.CasRelExporter,
    'spert': exporter.SpertExporter,
    'doc-red': exporter.DocRedExporter,
    'mrn': exporter.DocRedExporter,
    'mare': exporter.LineJsonExporter,
    'dual-dec': exporter.DualDecExporter,
    'ptr-net': exporter.PtrNetExporter
}


def main(input_paths: typing.List[str], export_path: str,
         import_fn: importer.BaseImporter, export_fn: exporter.BaseExporter,
         splits: typing.List[split.Split] = None, shuffle: bool = False, stratified: bool = False):
    data = import_fn.load(input_paths)

    data_valid = True
    sanity_checks: typing.List[typing.Union[sanity.BaseGlobalSanityCheck, sanity.BaseLocalSanityCheck]] = [
        sanity.TokenBoundaryCheck(),
        sanity.TokenIndicesCheck(),
        sanity.EntityFilledCheck()
    ]
    for check in sanity_checks:
        if isinstance(check, sanity.BaseGlobalSanityCheck):
            try:
                check.holds(data)
            except sanity.GlobalSanityCheckError as e:
                click.secho(f'Sanity check failed for {len(e.samples)} samples ({[s.id for s in e.samples]}) '
                            f'because of the following reason:', fg='red')
                click.secho(str(e), fg='red')
                click.secho('-' * 100, fg='red')
                data_valid = False
        elif isinstance(check, sanity.BaseLocalSanityCheck):
            for sample in data.samples:
                try:
                    check.holds(sample)
                except sanity.LocalSanityCheckError as e:
                    click.secho(f'Sanity check failed for sample with id {e.sample.id} '
                                f'because of the following reason:', fg='red')
                    click.secho(str(e), fg='red')
                    click.secho('-' * 100, fg='red')
                    data_valid = False
        else:
            raise AssertionError(f'Unknown check of type {type(check).__name__}')

    if not data_valid:
        click.secho('Data was invalid, not writing to disk', fg='red')
        return

    click.secho(f'Data passed {len(sanity_checks)} sanity check{"s" if len(sanity_checks) > 1 else ""}',
                fg='green')

    if shuffle:
        click.echo(f'Shuffling dataset.')
        random.shuffle(data.samples)

    split_data: typing.List[typing.Tuple[model.DataSet, str]] = [(data, export_path)]
    if splits is not None and len(splits) > 0:
        click.echo(f'Splitting dataset into {len(splits)} subsets.')
        split_data = [(d, os.path.join(export_path, s.name)) for d, s in split.apply_splits(splits, data, stratified)]

    click.echo('Writing data to disk...')
    for dataset, path in split_data:
        export_fn.save(dataset, path)
    click.echo('Done!')
