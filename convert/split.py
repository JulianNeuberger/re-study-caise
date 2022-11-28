import dataclasses
import typing

import click

import model


@dataclasses.dataclass
class AutoSplit:
    name: str


@dataclasses.dataclass
class Split:
    name: str
    ratio: float


def resolve_splits(splits: typing.List[typing.Union[Split, AutoSplit]]) -> typing.List[Split]:
    fixed_ratio = sum([s.ratio for s in splits if type(s) == Split])
    assert fixed_ratio >= 0.0
    assert fixed_ratio <= 1.0
    num_auto_splits = len([s for s in splits if type(s) == AutoSplit])
    return [
        s if type(s) == Split
        else Split(s.name, (1.0 - fixed_ratio) / num_auto_splits)
        for s in splits
    ]


def apply_splits(splits: typing.List[Split], dataset: model.DataSet,
                 stratified: bool = False) -> typing.List[typing.Tuple[model.DataSet, Split]]:

    if stratified:
        datasets_by_split = {}
        for s in splits:
            datasets_by_split[s.name] = model.DataSet(samples=[], name=s.name)

        samples_by_relation_type = {}
        for s in dataset.samples:
            assert len(s.relations) == 1, 'Can only do stratified splits for samples with exactly one relation.'
            relation_type = s.relations[0].type
            if relation_type not in samples_by_relation_type:
                samples_by_relation_type[relation_type] = []
            samples_by_relation_type[relation_type].append(s)

        for relation_type, samples in samples_by_relation_type.items():
            num_samples = len(samples)
            start = 0.0
            for s in splits:
                stop = start + s.ratio * num_samples
                datasets_by_split[s.name].samples.extend(samples[int(start):int(stop)])
                start = stop
        return list(zip(datasets_by_split.values(), splits))
    else:
        processed_datasets: typing.List[typing.Tuple[model.DataSet, Split]] = []
        num_samples = len(dataset.samples)
        start = 0.0
        for s in splits:
            stop = start + s.ratio * num_samples
            processed_dataset = model.DataSet(samples=dataset.samples[int(start):int(stop)], name=s.name)
            processed_datasets.append((processed_dataset, s))
            start = stop
        assert num_samples == start, f'Expected to have processed {num_samples} samples, but only processed {start}.'
        return processed_datasets


class SplitParam(click.ParamType):
    name = 'split'

    def convert(self,
                value: typing.Any,
                param: typing.Optional[click.Parameter],
                ctx: typing.Optional[click.Context]) -> typing.Any:
        if type(value) is str:
            if ':' not in value:
                self.fail(f'Unsupported format for split, "{value}" is missing a ":".')
            values: typing.List[str] = value.split(':')
            if len(values) != 2:
                self.fail(f'Unsupported format for split, expected "{value}" to have exactly one ":".')
            file_name: str
            raw_split_ratio: str
            file_name, raw_split_ratio = values
            if raw_split_ratio == '*':
                return AutoSplit(file_name)
            try:
                split_ratio = float(raw_split_ratio)
            except ValueError:
                self.fail(f'Expected value right of ":" to be either "*" '
                          f'or a valid floating point number, but got "{raw_split_ratio}".')
            if split_ratio <= 0.0 or split_ratio >= 1.0:
                self.fail(f'Expected split ratio to be in range ]0.0;1.0[, but got "{split_ratio}".')
            return Split(file_name, split_ratio)
        if type(value) is AutoSplit:
            return value
        if type(value) is Split:
            return value
        self.fail(f'Unsupported type "{type(value)}" of raw value.')
