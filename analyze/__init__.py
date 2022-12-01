import os
import re
import typing

import matplotlib
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
import tabulate
import unicodedata
from matplotlib import pyplot as plt

import model
from analyze import base, entity
from analyze import structure
from convert import importer

matplotlib.rcParams.update({'figure.autolayout': True})


def run_analysis(datasets: typing.List[model.DataSet], steps: typing.List[base.AnalysisStep]):
    analysis_results: typing.Dict[str, typing.List[base.AnalysisResult]] = {
        d.name: [] for d in datasets
    }

    for step in steps:
        for dataset in datasets:
            results = step.calculate_results(dataset)
            analysis_results[dataset.name].extend(results)
    return analysis_results


def print_tabular(datasets: typing.List[model.DataSet],
                  steps: typing.List[base.AnalysisStep],
                  table_fmt: str = None,
                  save_path: str = None):
    analysis_results = run_analysis(datasets, steps)

    headers = ['dataset']
    for dataset_name, analysis_result in analysis_results.items():
        for step_result in analysis_result:
            # dont use a set, as order matters
            if step_result.id not in headers:
                headers.append(step_result.id)

    rows: typing.List[typing.Dict[str, str]] = []
    for dataset_name, analysis_result in analysis_results.items():
        row = {}
        for result_name in headers:
            row[result_name] = ''
        row['dataset'] = dataset_name

        for step_result in analysis_result:
            row[step_result.id] = str(step_result.value)
        rows.append(row)

    content = [
        [row[header] for header in headers]
        for row in rows
    ]

    tabular_data = tabulate.tabulate(content, headers=headers, tablefmt=table_fmt, floatfmt='.2f')
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(tabular_data)
    else:
        print(tabular_data)


def analysis_step_by_name(results: typing.List[base.AnalysisResult], name: str) -> base.AnalysisResult:
    for r in results:
        if r.id == name:
            return r
    raise ValueError(f'Result has no key {name}')


def setup_axes(ax: plt.Axes = None):
    if ax is None:
        plt.xticks(rotation=90)
    else:
        ax.set_xticklabels(ax.get_xticks(), rotation=90)


def setup_figure(fig_size: typing.Tuple[float, float] = None,
                 title: str = None):
    if fig_size is not None:
        plt.figure(figsize=fig_size)
    else:
        plt.figure()
    if title is not None:
        plt.title(title)


def setup_labels(x_label: str = None, y_label: str = None):
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)


def finalize_figure(save_paths: typing.List[str] = None):
    if save_paths is not None:
        for save_path in save_paths:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def print_bar_plot_per_dataset(datasets: typing.List[model.DataSet],
                               steps: typing.List[base.AnalysisStep],
                               get_value: typing.Callable[[typing.List[base.AnalysisResult]], float],
                               fig_size: typing.Tuple[float, float] = None,
                               x_label: str = None,
                               y_label: str = None,
                               title: str = None,
                               save_paths: typing.List[str] = None):
    sns.set_theme()

    analysis_results = run_analysis(datasets, steps)

    cols = {
        'keys': list(analysis_results.keys()),
        'values': [get_value(r) for r in analysis_results.values()]
    }

    setup_figure(fig_size, title)
    setup_axes()
    dataframe = pd.DataFrame(data=cols)
    sns.barplot(x='keys', y='values', data=dataframe)
    setup_labels(x_label, y_label)
    finalize_figure(save_paths)


def print_bar_plot_per_step(dataset: model.DataSet,
                            step: base.AnalysisStep,
                            get_cols: typing.Callable[[typing.List[base.AnalysisResult]],
                                                      typing.Tuple[typing.List, typing.List]],
                            fig_size: typing.Tuple[float, float] = None,
                            on_axes: plt.Axes = None,
                            show_x_ticks: bool = True,
                            x_label: str = None,
                            y_label: str = None,
                            title: str = None,
                            should_finalize: bool = True,
                            save_paths: typing.List[str] = None,
                            color: typing.Any = None):
    sns.set_theme()

    if on_axes is not None:
        assert fig_size is None
        assert title is None

    analysis_results = run_analysis([dataset], [step])

    cols = get_cols(analysis_results[dataset.name])
    cols = {
        'keys': cols[0],
        'values': cols[1]
    }

    if on_axes is None:
        setup_figure(fig_size, title)
    setup_axes(on_axes)
    dataframe = pd.DataFrame(data=cols)
    g = sns.barplot(x='keys', y='values', data=dataframe, ax=on_axes, color=color)
    if not show_x_ticks:
        g.set(xticklabels=[])

    if on_axes is None:
        setup_labels(x_label, y_label)
    else:
        on_axes.set(xlabel=x_label, ylabel=y_label)
    if should_finalize:
        finalize_figure(save_paths)


def print_heatmap_per_step(dataset: model.DataSet,
                           step: base.AnalysisStep,
                           get_dataframe: typing.Callable[[typing.List[base.AnalysisResult]], pd.DataFrame],
                           fig_size: typing.Tuple[float, float] = None,
                           x_label: str = None,
                           y_label: str = None,
                           title: str = None,
                           save_paths: typing.List[str] = None,
                           annotation_format: str = '.0f'):
    sns.set_theme()

    analysis_results = run_analysis([dataset], [step])
    matrix = get_dataframe(analysis_results[dataset.name])

    setup_figure(fig_size, title)
    setup_axes()
    sns.heatmap(matrix, annot=True, fmt=annotation_format, xticklabels=True, yticklabels=True)
    setup_labels(x_label, y_label)
    finalize_figure(save_paths)


def group_datasets(datasets: typing.Iterable[model.DataSet], group_name: str) -> model.DataSet:
    samples = []
    for d in datasets:
        samples.extend(d.samples)
    return model.DataSet(samples=samples, name=group_name)


def cols_from_relation_types(relation_types: typing.Dict[str, int]) -> typing.Tuple[typing.List[str],
                                                                                    typing.List[float]]:
    relation_names = []
    relation_counts = []
    sorted_items = sorted(relation_types.items(), key=lambda item: item[1], reverse=True)
    for name, count in sorted_items:
        relation_names.append(name)
        relation_counts.append(count)
    return relation_names, relation_counts


def cols_from_num_relations_per_sample(num_relations_per_sample: typing.Dict[int, int]) -> typing.Tuple[typing.List[int],
                                                                                                        typing.List[int]]:
    return list(num_relations_per_sample.keys()), list(num_relations_per_sample.values())


def dataframe_from_dicts(dictionary: typing.Dict[str, typing.Dict[str, int]],
                         row_threshold: float = 10.0, col_threshold: float = 0.0) -> pandas.DataFrame:
    x_labels = list(dictionary.keys())
    y_labels = set()
    for inner_dict in dictionary.values():
        y_labels.update(inner_dict.keys())
    y_labels = list(y_labels)

    matrix = np.zeros(shape=(len(y_labels), len(x_labels)))

    for x_label, inner_dict in dictionary.items():
        for y_label, num_co_occurrences in inner_dict.items():
            matrix[y_labels.index(y_label), x_labels.index(x_label)] = num_co_occurrences

    dataframe = pandas.DataFrame(
        matrix,
        columns=x_labels,
        index=y_labels
    )

    # sort by column names, i.e. the relation types
    dataframe = dataframe.sort_index(axis=1)
    dataframe = dataframe.sort_values(dataframe.columns.tolist(), ascending=False)

    # filter out rows (i.e. ner tag combinations) that fall under a threshold
    dataframe = dataframe[dataframe.sum(axis=1) > row_threshold]

    # filter out cols that fall under a threshold
    col_totals = dataframe.sum(axis=0)
    dataframe = dataframe[col_totals[col_totals > col_threshold].index]

    # remove 0 values
    dataframe[dataframe == 0] = np.NaN

    return dataframe


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


if __name__ == '__main__':
    print('Loading data...')
    nyt_data = [
        importer.JsonLinesImporter().load(['../data/export/nyt10/jsonl/train.json'], name='NYT10 (train)'),
        importer.JsonLinesImporter().load(['../data/export/nyt10/jsonl/valid.json'], name='NYT10 (dev/valid)'),
        importer.JsonLinesImporter().load(['../data/export/nyt10/jsonl/test.json'], name='NYT10 (test)'),
    ]
    semeval_data = [
        importer.JsonLinesImporter().load(['../data/export/semeval/jsonl/train.json'], name='Semeval (train)'),
        importer.JsonLinesImporter().load(['../data/export/semeval/jsonl/valid.json'], name='Semeval (dev/valid)'),
        importer.JsonLinesImporter().load(['../data/export/semeval/jsonl/test.json'], name='Semeval (test)'),
    ]
    conll_data = [
        importer.JsonLinesImporter().load(['../data/export/conll04/jsonl/train.json'], name='ConLL 04 (train)'),
        importer.JsonLinesImporter().load(['../data/export/conll04/jsonl/valid.json'], name='ConLL 04 (dev/valid)'),
        importer.JsonLinesImporter().load(['../data/export/conll04/jsonl/test.json'], name='ConLL 04 (test)'),
    ]
    fewrel_data = [
        importer.JsonLinesImporter().load(['../data/export/fewrel/jsonl/train.json'], name='FewRel (train)'),
        importer.JsonLinesImporter().load(['../data/export/fewrel/jsonl/valid.json'], name='FewRel (dev/valid)'),
        importer.JsonLinesImporter().load(['../data/export/fewrel/jsonl/test.json'], name='FewRel (test)'),
    ]

    grouped_nyt_data = group_datasets(group_name='NYT10', datasets=nyt_data)
    grouped_semeval_data = group_datasets(group_name='SemEval 2010 (task 8)', datasets=semeval_data)
    grouped_conll_data = group_datasets(group_name='ConLL 04', datasets=conll_data)
    grouped_fewrel_data = group_datasets(group_name='FewRel', datasets=fewrel_data)

    data = [grouped_nyt_data, grouped_semeval_data, grouped_conll_data, grouped_fewrel_data]

    dataset_names = {d.name for d in data}
    assert len(dataset_names) == len(data), f'Dataset names have to be unique for analysis purposes.'

    palette = sns.color_palette()

    print_tabular(
        datasets=nyt_data + [grouped_nyt_data] + semeval_data + [grouped_semeval_data] + conll_data + [grouped_conll_data] + fewrel_data + [grouped_fewrel_data],
        steps=[
            structure.NumSamplesAnalysisStep(),
            structure.RelationTypesAnalysisStep(),
            structure.RelationDistanceAnalysisStep(),
            structure.VocabSizeAnalysisStep(),
            structure.NumEntityAnalysisStep()
        ],
        table_fmt='latex_raw',
        save_path=os.path.join('export', 'detailed_dataset_overview.tex')
    )

    # Number of examples per dataset
    print_bar_plot_per_dataset(
        datasets=data,
        steps=[structure.NumSamplesAnalysisStep()],
        get_value=lambda r: analysis_step_by_name(r, 'num_samples').value,
        x_label='dataset name',
        y_label='number of examples',
        save_paths=[os.path.join('export', 'num_samples.png'), os.path.join('export', 'num_samples.pdf')]
    )

    # Number of unique relation types per dataset
    # print_bar_plot_per_dataset(
    #     datasets=data,
    #     steps=[structure.RelationTypesAnalysisStep()],
    #     get_value=lambda r: analysis_step_by_name(r, 'num_relation_types').value,
    #     x_label='dataset name',
    #     y_label='number of unique relation types',
    #     save_paths=[os.path.join('export', 'num_relations.png'), os.path.join('export', 'num_relations.pdf')]
    # )

    # average token distance between relation entity heads
    # print_bar_plot_per_dataset(
    #     datasets=data,
    #     steps=[structure.RelationDistanceAnalysisStep()],
    #     get_value=lambda r: analysis_step_by_name(r, 'avg_relation_distance').value,
    #     x_label='dataset name',
    #     y_label='distance in tokens',
    #     save_paths=[os.path.join('export', 'relation_distance.png'), os.path.join('export', 'relation_distance.pdf')]
    # )

    # average number of tokens per entity
    print_bar_plot_per_dataset(
        datasets=data,
        steps=[structure.EntityLengthAnalysisStep()],
        get_value=lambda r: analysis_step_by_name(r, 'avg_entity_length').value,
        x_label='dataset name',
        y_label='average entity length in tokens',
        save_paths=[os.path.join('export', 'entity_length.png'), os.path.join('export', 'entity_length.pdf')]
    )

    # number of ner tags
    print_bar_plot_per_dataset(
        datasets=data,
        steps=[structure.NerTagAnalysisStep()],
        get_value=lambda r: analysis_step_by_name(r, 'num_ner_tags').value,
        x_label='dataset name',
        y_label='number of NER tags',
        save_paths=[os.path.join('export', 'num_ner_tags.png'), os.path.join('export', 'num_ner_tags.pdf')]
    )

    # Relation distribution small version
    fig, ax = plt.subplots(ncols=len(data), figsize=(18.0, 4.8), gridspec_kw={'width_ratios': [24, 9, 5, 71]})
    fig.supylabel('number of relation types')
    for i, d in enumerate(data):
        print_bar_plot_per_step(
            dataset=d,
            step=structure.RelationTypesAnalysisStep(return_relation_count=['all']),
            get_cols=lambda r: cols_from_relation_types(analysis_step_by_name(r, 'relation_types').value),
            on_axes=ax[i],
            x_label=d.name,
            should_finalize=False,
            show_x_ticks=False,
            color=palette[0]
        )
    finalize_figure(save_paths=[os.path.join('export', f'relation_distributions_small.png'),
                                os.path.join('export', f'relation_distributions_small.pdf')])

    # Relation distribution large version
    fig = plt.figure(figsize=(18.0, 14.0))
    gs = fig.add_gridspec(2, 4)
    axs = [
        fig.add_subplot(gs[0, :2]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),

        fig.add_subplot(gs[1, :])
    ]

    # fig, ax = plt.subplots(nrows=2, ncols=len(data) - 1, figsize=(18.0, 6.8), gridspec_kw={'width_ratios': [24, 9, 5, 71]})
    for i, d in enumerate(data):
        print_bar_plot_per_step(
            dataset=d,
            step=structure.RelationTypesAnalysisStep(return_relation_count=['all']),
            get_cols=lambda r: cols_from_relation_types(analysis_step_by_name(r, 'relation_types').value),
            on_axes=axs[i],
            should_finalize=False,
            color=palette[0]
        )
    finalize_figure(save_paths=[os.path.join('export', f'relation_distributions.png'),
                                os.path.join('export', f'relation_distributions.pdf')])

    # relation distribution single versions
    fig_sizes = [(12.8, 6.8), (4.5, 4.0), (3.0, 3.8), (25.6, 6.8)]
    for i, d in enumerate(data):
        plt.figure(figsize=fig_sizes[i])
        print_bar_plot_per_step(
            dataset=d,
            step=structure.RelationTypesAnalysisStep(return_relation_count=['all']),
            get_cols=lambda r: cols_from_relation_types(analysis_step_by_name(r, 'relation_types').value),
            should_finalize=False,
            on_axes=plt.gca(),
            color=palette[0]
        )
        finalize_figure(save_paths=[os.path.join('export', f'relation_distributions_{slugify(d.name)}.png'),
                                    os.path.join('export', f'relation_distributions_{slugify(d.name)}.pdf')])

    # Share of negative examples
    for d in data:
        print_bar_plot_per_step(
            dataset=d,
            step=structure.NegativeExamplesAnalysisStep(),
            get_cols=lambda r: (['no relations', 'at least one relation'],
                                [analysis_step_by_name(r, 'negative_examples').value,
                                 analysis_step_by_name(r, 'positive_examples').value]),
            x_label='',
            y_label='number of examples',
            save_paths=[os.path.join('export', f'negative_examples_{slugify(d.name)}.png'),
                        os.path.join('export', f'negative_examples_{slugify(d.name)}.pdf')]
        )

    for d in data:
        print_bar_plot_per_step(
            dataset=d,
            step=structure.NumRelationsPerSampleAnalysisStep(),
            get_cols=lambda r: cols_from_num_relations_per_sample(analysis_step_by_name(r, 'num_relations_per_sample').value),
            x_label='relations per sample',
            y_label='number of samples',
            save_paths=[os.path.join('export', f'num_relations_per_sample_{slugify(d.name)}.png'),
                        os.path.join('export', f'num_relations_per_sample_{slugify(d.name)}.pdf')]
        )

    # heatmap for ner tag co-occurrence in relation head and tails
    print_heatmap_per_step(
        dataset=grouped_conll_data,
        step=structure.RelationEntityTagAnalysisStep(),
        get_dataframe=lambda r: dataframe_from_dicts(analysis_step_by_name(r, 'relation_entity_tags').value, row_threshold=0.0),
        fig_size=(10, 10),
        x_label='',
        y_label='',
        save_paths=[os.path.join('export', f'relation_entity_tags_{slugify(grouped_conll_data.name)}.png'),
                    os.path.join('export', f'relation_entity_tags_{slugify(grouped_conll_data.name)}.pdf')]
    )

    print_heatmap_per_step(
        dataset=grouped_semeval_data,
        step=structure.RelationEntityTagAnalysisStep(),
        get_dataframe=lambda r: dataframe_from_dicts(analysis_step_by_name(r, 'relation_entity_tags').value, row_threshold=0.0),
        fig_size=(10, 18),
        x_label='',
        y_label='',
        save_paths=[os.path.join('export', f'relation_entity_tags_{slugify(grouped_semeval_data.name)}.png'),
                   os.path.join('export', f'relation_entity_tags_{slugify(grouped_semeval_data.name)}.pdf')]
    )

    print_heatmap_per_step(
        dataset=grouped_nyt_data,
        step=structure.RelationEntityTagAnalysisStep(),
        get_dataframe=lambda r: dataframe_from_dicts(analysis_step_by_name(r, 'relation_entity_tags').value, row_threshold=20.0),
        fig_size=(20, 20),
        x_label='',
        y_label='',
        save_paths=[os.path.join('export', f'relation_entity_tags_{slugify(grouped_nyt_data.name)}.png'),
                   os.path.join('export', f'relation_entity_tags_{slugify(grouped_nyt_data.name)}.pdf')]
    )

    print_heatmap_per_step(
        dataset=grouped_fewrel_data,
        step=structure.RelationEntityTagAnalysisStep(),
        get_dataframe=lambda r: dataframe_from_dicts(analysis_step_by_name(r, 'relation_entity_tags').value, row_threshold=20.0),
        fig_size=(20, 20),
        x_label='',
        y_label='',
        save_paths=[os.path.join('export', f'relation_entity_tags_{slugify(grouped_fewrel_data.name)}.png'),
                   os.path.join('export', f'relation_entity_tags_{slugify(grouped_fewrel_data.name)}.pdf')]
    )

