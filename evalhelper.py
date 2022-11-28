import collections
import csv
import dataclasses
import os

import matplotlib
import numpy as np
import pandas as pd
import seaborn
import seaborn as sns
import tabulate
import typing
from matplotlib import pyplot as plt
from matplotlib import lines
from scipy import stats

import convert
import metrics
import model
from metrics import parser, binning


@dataclasses.dataclass
class DatasetResult:
    id: str
    name: str

    test_results: typing.List[str]
    valid_results: typing.List[str]


@dataclasses.dataclass
class ApproachResult:
    id: str
    name: str

    datasets: typing.List[DatasetResult]


base_result_dir = os.path.join('metrics', 'results')

results = [
    ApproachResult(
        id='casrel',
        name='CasRel',
        datasets=[
            DatasetResult(id='conll04', name='ConLL 04',
                          test_results=[os.path.join(base_result_dir, 'casrel', 'ai4-conll04', 'test_result.json')],
                          valid_results=[os.path.join(base_result_dir, 'casrel', 'ai4-conll04', 'dev_result.json')]),
            DatasetResult(id='fewrel', name='FewRel',
                          test_results=[os.path.join(base_result_dir, 'casrel', 'ai4-fewrel', 'test_result.json')],
                          valid_results=[os.path.join(base_result_dir, 'casrel', 'ai4-fewrel', 'dev_result.json')]),
            DatasetResult(id='nyt10', name='NYT10',
                          test_results=[os.path.join(base_result_dir, 'casrel', 'ai4-nyt10', 'test_result.json')],
                          valid_results=[os.path.join(base_result_dir, 'casrel', 'ai4-nyt10', 'dev_result.json')]),
            DatasetResult(id='semeval', name='Semeval',
                          test_results=[os.path.join(base_result_dir, 'casrel', 'ai4-semeval', 'test_result.json')],
                          valid_results=[os.path.join(base_result_dir, 'casrel', 'ai4-semeval', 'dev_result.json')]),
        ]),

    ApproachResult(
        id='joint-er',
        name='JointER',
        datasets=[
            DatasetResult(id='conll04', name='ConLL 04',
                          test_results=[os.path.join(base_result_dir, 'joint-er', 'conll04', 'best_test_results.json')],
                          valid_results=[
                              os.path.join(base_result_dir, 'joint-er', 'conll04', 'best_dev_results.json')]),
            DatasetResult(id='fewrel', name='FewRel',
                          test_results=[os.path.join(base_result_dir, 'joint-er', 'fewrel', 'best_test_results.json')],
                          valid_results=[os.path.join(base_result_dir, 'joint-er', 'fewrel', 'best_dev_results.json')]),
            DatasetResult(id='nyt10', name='NYT10',
                          test_results=[os.path.join(base_result_dir, 'joint-er', 'nyt10', 'best_test_results.json')],
                          valid_results=[os.path.join(base_result_dir, 'joint-er', 'nyt10', 'best_dev_results.json')]),
            DatasetResult(id='semeval', name='Semeval',
                          test_results=[os.path.join(base_result_dir, 'joint-er', 'semeval', 'best_test_results.json')],
                          valid_results=[
                              os.path.join(base_result_dir, 'joint-er', 'semeval', 'best_dev_results.json')]),
        ]),

    ApproachResult(
        id='rsan',
        name='RSAN',
        datasets=[
            DatasetResult(id='conll04', name='ConLL 04',
                          test_results=[os.path.join(base_result_dir, 'RSAN', 'conll04', 'test-results.json')],
                          valid_results=[os.path.join(base_result_dir, 'RSAN', 'conll04', 'dev-results.json')]),
            DatasetResult(id='fewrel', name='FewRel',
                          test_results=[os.path.join(base_result_dir, 'RSAN', 'fewrel', 'test-results.json')],
                          valid_results=[os.path.join(base_result_dir, 'RSAN', 'fewrel', 'dev-results.json')]),
            DatasetResult(id='nyt10', name='NYT10',
                          test_results=[os.path.join(base_result_dir, 'RSAN', 'nyt10', 'test-results.json')],
                          valid_results=[os.path.join(base_result_dir, 'RSAN', 'nyt10', 'dev-results.json')]),
            DatasetResult(id='semeval', name='Semeval',
                          test_results=[os.path.join(base_result_dir, 'RSAN', 'semeval', 'test-results.json')],
                          valid_results=[os.path.join(base_result_dir, 'RSAN', 'semeval', 'dev-results.json')]),
        ]),

    ApproachResult(
        id='spert',
        name='SpERT',
        datasets=[
            DatasetResult(id='conll04', name='ConLL 04',
                          test_results=[os.path.join('data', 'export', 'conll04', 'jsonl', 'test.json'),
                                        os.path.join(base_result_dir, 'spert', 'conll04',
                                                     'predictions_test_epoch_0.json')],
                          valid_results=[os.path.join('data', 'export', 'conll04', 'jsonl', 'valid.json'),
                                         os.path.join(base_result_dir, 'spert', 'conll04',
                                                      'predictions_dev_epoch_0.json')]),
            DatasetResult(id='fewrel', name='FewRel',
                          test_results=[os.path.join('data', 'export', 'fewrel', 'jsonl', 'test.json'),
                                        os.path.join(base_result_dir, 'spert', 'fewrel',
                                                     'predictions_test_epoch_0.json')],
                          valid_results=[os.path.join('data', 'export', 'fewrel', 'jsonl', 'valid.json'),
                                         os.path.join(base_result_dir, 'spert', 'fewrel',
                                                      'predictions_dev_epoch_0.json')]),
            DatasetResult(id='nyt10', name='NYT10',
                          test_results=[os.path.join('data', 'export', 'nyt10', 'jsonl', 'test.json'),
                                        os.path.join(base_result_dir, 'spert', 'nyt10',
                                                     'predictions_test_epoch_0.json')],
                          valid_results=[os.path.join('data', 'export', 'nyt10', 'jsonl', 'valid.json'),
                                         os.path.join(base_result_dir, 'spert', 'nyt10',
                                                      'predictions_dev_epoch_0.json')]),
            DatasetResult(id='semeval', name='Semeval',
                          test_results=[os.path.join('data', 'export', 'semeval', 'jsonl', 'test.json'),
                                        os.path.join(base_result_dir, 'spert', 'semeval',
                                                     'predictions_test_epoch_0.json')],
                          valid_results=[os.path.join('data', 'export', 'semeval', 'jsonl', 'valid.json'),
                                         os.path.join(base_result_dir, 'spert', 'semeval',
                                                      'predictions_dev_epoch_0.json')]),
        ]),

    ApproachResult(
        id='two-are-better-than-one',
        name='Two',
        datasets=[
            DatasetResult(id='conll04', name='ConLL 04',
                          test_results=[os.path.join(base_result_dir, 'two-are-better-than-one', 'conll04',
                                                     'model.results-test.json')],
                          valid_results=[os.path.join(base_result_dir, 'two-are-better-than-one', 'conll04',
                                                      'model.results-valid.json')]),
            DatasetResult(id='fewrel', name='FewRel',
                          test_results=[os.path.join(base_result_dir, 'two-are-better-than-one', 'fewrel',
                                                     'model.results-test.json')],
                          valid_results=[os.path.join(base_result_dir, 'two-are-better-than-one', 'fewrel',
                                                      'model.results-valid.json')]),
            DatasetResult(id='nyt10', name='NYT10',
                          test_results=[os.path.join(base_result_dir, 'two-are-better-than-one', 'nyt10',
                                                     'model.results-test.json')],
                          valid_results=[os.path.join(base_result_dir, 'two-are-better-than-one', 'nyt10',
                                                      'model.results-valid.json')]),
            DatasetResult(id='semeval', name='Semeval',
                          test_results=[os.path.join(base_result_dir, 'two-are-better-than-one', 'semeval',
                                                     'model.results-test.json')],
                          valid_results=[os.path.join(base_result_dir, 'two-are-better-than-one', 'semeval',
                                                      'model.results-valid.json')]),
        ]),

    ApproachResult(
        id='pfn',
        name='PFN',
        datasets=[
            DatasetResult(id='conll04', name='ConLL 04',
                          test_results=[os.path.join('data', 'export', 'conll04', 'pfn', 'test.json'),
                                        os.path.join(base_result_dir, 'PFN-custom', 'result-conll04-test.json')],
                          valid_results=[os.path.join('data', 'export', 'conll04', 'pfn', 'valid.json'),
                                         os.path.join(base_result_dir, 'PFN-custom', 'result-conll04-dev.json')]),
            DatasetResult(id='fewrel', name='FewRel',
                          test_results=[os.path.join('data', 'export', 'fewrel', 'pfn', 'test.json'),
                                        os.path.join(base_result_dir, 'PFN-custom', 'result-fewrel-test.json')],
                          valid_results=[os.path.join('data', 'export', 'fewrel', 'pfn', 'valid.json'),
                                         os.path.join(base_result_dir, 'PFN-custom', 'result-fewrel-dev.json')]),
            DatasetResult(id='nyt10', name='NYT10',
                          test_results=[os.path.join('data', 'export', 'nyt10', 'pfn', 'test.json'),
                                        os.path.join(base_result_dir, 'PFN-custom', 'result-nyt10-test.json')],
                          valid_results=[os.path.join('data', 'export', 'nyt10', 'pfn', 'valid.json'),
                                         os.path.join(base_result_dir, 'PFN-custom', 'result-nyt10-dev.json')]),
            DatasetResult(id='semeval', name='Semeval',
                          test_results=[os.path.join('data', 'export', 'semeval', 'pfn', 'test.json'),
                                        os.path.join(base_result_dir, 'PFN-custom', 'result-semeval-test.json')],
                          valid_results=[os.path.join('data', 'export', 'semeval', 'pfn', 'valid.json'),
                                         os.path.join(base_result_dir, 'PFN-custom', 'result-semeval-dev.json')]),
        ]),

    ApproachResult(
        id='mare',
        name='MARE',
        datasets=[
            DatasetResult(id='conll04', name='ConLL 04',
                          test_results=[os.path.join('data', 'export', 'conll04', 'jsonl', 'test.json'),
                                        os.path.join(base_result_dir, 'mare', 'ai4-conll04-test.json')],
                          valid_results=[os.path.join('data', 'export', 'conll04', 'jsonl', 'valid.json'),
                                         os.path.join(base_result_dir, 'mare', 'ai4-conll04-dev.json')]),
            DatasetResult(id='fewrel', name='FewRel',
                          test_results=[os.path.join('data', 'export', 'fewrel', 'jsonl', 'test.json'),
                                        os.path.join(base_result_dir, 'mare', 'ai4-fewrel-test.json')],
                          valid_results=[os.path.join('data', 'export', 'fewrel', 'jsonl', 'valid.json'),
                                         os.path.join(base_result_dir, 'mare', 'ai4-fewrel-dev.json')]),
            DatasetResult(id='nyt10', name='NYT10',
                          test_results=[os.path.join('data', 'export', 'nyt10', 'jsonl', 'test.json'),
                                        os.path.join(base_result_dir, 'mare', 'ai4-nyt10-test.json')],
                          valid_results=[os.path.join('data', 'export', 'nyt10', 'jsonl', 'valid.json'),
                                         os.path.join(base_result_dir, 'mare', 'ai4-nyt10-dev.json')]),
            DatasetResult(id='semeval', name='Semeval',
                          test_results=[os.path.join('data', 'export', 'semeval', 'jsonl', 'test.json'),
                                        os.path.join(base_result_dir, 'mare', 'ai4-semeval-test.json')],
                          valid_results=[os.path.join('data', 'export', 'semeval', 'jsonl', 'valid.json'),
                                         os.path.join(base_result_dir, 'mare', 'ai4-semeval-dev.json')]),
        ]),
]

sns.set_theme()
matplotlib.rcParams.update({'figure.autolayout': True})

_results = {
    'casrel': {
        'nyt10': [os.path.join(base_result_dir, 'casrel', 'ai4-nyt10', 'test_result.json')],
        'conll04': [os.path.join(base_result_dir, 'casrel', 'ai4-conll04', 'test_result.json')],
        'semeval': [os.path.join(base_result_dir, 'casrel', 'ai4-semeval', 'test_result.json')],
    },
    'joint-er': {
        'nyt10': [os.path.join(base_result_dir, 'joint-er', 'nyt10', 'best_test_results.json')],
        'conll04': [os.path.join(base_result_dir, 'joint-er', 'conll04', 'best_test_results.json')],
        'semeval': [os.path.join(base_result_dir, 'joint-er', 'semeval', 'best_test_results.json')],
    },
    'rsan': {
        'nyt10': [os.path.join(base_result_dir, 'RSAN', 'nyt10', 'results.json')],
        'conll04': [os.path.join(base_result_dir, 'RSAN', 'conll04', 'results.json')],
        'semeval': [os.path.join(base_result_dir, 'RSAN', 'semeval', 'results.json')],
    },
    'spert': {
        'nyt10': [os.path.join(base_result_dir, 'spert', 'nyt10', 'test.json'),
                  os.path.join(base_result_dir, 'spert', 'nyt10', 'predictions_test_epoch_0.json')],
        'conll04': [os.path.join(base_result_dir, 'spert', 'conll04', 'test.json'),
                    os.path.join(base_result_dir, 'spert', 'conll04', 'predictions_test_epoch_0.json')],
        'semeval': [os.path.join(base_result_dir, 'spert', 'semeval', 'test.json'),
                    os.path.join(base_result_dir, 'spert', 'semeval', 'predictions_test_epoch_0.json')],
    },
    'two-are-better-than-one': {
        'nyt10': [os.path.join(base_result_dir, 'two-are-better-than-one', 'nyt10', 'model.results.json')],
        'conll04': [os.path.join(base_result_dir, 'two-are-better-than-one', 'conll04', 'model.results.json')],
        'semeval': [os.path.join(base_result_dir, 'two-are-better-than-one', 'semeval', 'model.results.json')],
        'fewrel': [os.path.join(base_result_dir, 'two-are-better-than-one', 'fewrel', 'model.results-test.json')]
    },
}


@dataclasses.dataclass
class PlottingData:
    approaches: typing.List = dataclasses.field(default_factory=list)
    f1_scores: typing.List = dataclasses.field(default_factory=list)
    p_scores: typing.List = dataclasses.field(default_factory=list)
    r_scores: typing.List = dataclasses.field(default_factory=list)
    datasets: typing.List = dataclasses.field(default_factory=list)
    f1_types: typing.List = dataclasses.field(default_factory=list)
    confusion_matrices: typing.List[np.ndarray] = dataclasses.field(default_factory=list)
    n_ok_entities: typing.List[int] = dataclasses.field(default_factory=list)
    n_gold_entities: typing.List[int] = dataclasses.field(default_factory=list)
    labels: typing.List[typing.List[str]] = dataclasses.field(default_factory=list)
    eval_types: typing.List = dataclasses.field(default_factory=list)

    def add(self,
            approach: str, f1_score: float, p_score: float, r_score: float,
            dataset: str, f1_type: str, confusion_matrix: np.ndarray,
            n_ok_entities: int, n_gold_entities: int,
            labels: typing.List[str], eval_type: str,
            *args, **kwargs):
        self.approaches.append(approach)
        self.f1_scores.append(f1_score)
        self.p_scores.append(p_score)
        self.r_scores.append(r_score)
        self.f1_types.append(f1_type)
        self.datasets.append(dataset)
        self.confusion_matrices.append(confusion_matrix)
        self.n_gold_entities.append(n_gold_entities)
        self.n_ok_entities.append(n_ok_entities)
        self.labels.append(labels)
        self.eval_types.append(eval_type)

    def to_df(self):
        return pd.DataFrame({
            'approach': self.approaches,
            'f1': self.f1_scores,
            'p': self.p_scores,
            'r': self.r_scores,
            'f1-type': self.f1_types,
            'dataset': self.datasets,
            'eval_type': self.eval_types,
            'n_ok': self.n_ok_entities,
            'n_gold': self.n_gold_entities
        })


@dataclasses.dataclass
class BinnedPlottingData(PlottingData):
    bin_ids: typing.List[int] = dataclasses.field(default_factory=list)
    bin_values: typing.List[float] = dataclasses.field(default_factory=list)
    bin_counts: typing.List[int] = dataclasses.field(default_factory=list)

    def add(self,
            approach: str, f1_score: float, p_score: float, r_score: float,
            dataset: str, f1_type: str, confusion_matrix: np.ndarray,
            n_ok_entities: int, n_gold_entities: int,
            labels: typing.List[str], eval_type: str, bin_id: int, bin_value: float, bin_count: int):
        super().add(approach, f1_score, p_score, r_score, dataset, f1_type, confusion_matrix,
                    n_ok_entities, n_gold_entities, labels, eval_type)
        self.bin_ids.append(bin_id)
        self.bin_values.append(bin_value)
        self.bin_counts.append(bin_count)

    def to_df(self):
        ret = super().to_df()
        ret['bin_id'] = self.bin_ids
        ret['bin_value'] = self.bin_values
        ret['bin_count'] = self.bin_counts
        return ret


@dataclasses.dataclass
class VariabilityPlottingData(PlottingData):
    variabilities: typing.List[float] = dataclasses.field(default_factory=list)
    relation_types: typing.List[str] = dataclasses.field(default_factory=list)

    def add(self,
            approach: str, f1_score: float, p_score: float, r_score: float,
            dataset: str, f1_type: str, confusion_matrix: np.ndarray,
            n_ok_entities: int, n_gold_entities: int,
            labels: typing.List[str], eval_type: str,
            variability: float, relation_type: str):
        super().add(approach, f1_score, p_score, r_score, dataset, f1_type, confusion_matrix,
                    n_ok_entities, n_gold_entities, labels, eval_type)
        self.variabilities.append(variability)
        self.relation_types.append(relation_type)

    def to_df(self):
        ret = super().to_df()
        ret['variability'] = self.variabilities
        ret['relation_type'] = self.relation_types
        return ret


def write_result_tables(data: PlottingData):
    os.makedirs('figures', exist_ok=True)

    data = data.to_df()
    for subset in ['test', 'valid']:
        subdata: pd.DataFrame = data[data['eval_type'] == subset]
        subdata = subdata.drop(columns=['eval_type', 'n_gold', 'n_ok'])
        subdata = subdata.melt(id_vars=['dataset', 'approach', 'f1-type'], var_name='metric', value_name='score')
        subdata = subdata.sort_values(by=['f1-type', 'metric'])
        subdata = subdata.pivot(index=['dataset', 'approach'], columns=['metric', 'f1-type'])
        subdata = subdata.reset_index(level='dataset')
        subdata = subdata.sort_values(by=['approach', 'dataset'])

        tabular_data = tabulate.tabulate(subdata, headers='keys', tablefmt='latex_raw', floatfmt='.2%')
        tabular_data = tabular_data.replace('%', r'\%')

        for approach_name in set(subdata.index.values):
            tabular_data = tabular_data.replace(
                approach_name,
                fr'\hline\multirow{{4}}{{*}}{{\rotatebox[origin=c]{{90}}{{{approach_name}}}}}',
                1
            )
            tabular_data = tabular_data.replace(
                f' {approach_name} ',
                '  ' + ' ' * len(approach_name)
            )

        with open(os.path.join('figures', f'{subset}-results.tex'), 'w') as f:
            f.write(tabular_data)


def plot_confusion_matrix(data: PlottingData, name: str):
    for confusion_matrix, approach_name, relation_tags, dataset_name, eval_type in zip(data.confusion_matrices,
                                                                                       data.approaches, data.labels,
                                                                                       data.datasets, data.eval_types):
        confusion_as_list = confusion_matrix.tolist()
        confusion_as_df = pd.DataFrame(confusion_as_list, index=relation_tags, columns=relation_tags)

        plt.figure()
        seaborn.heatmap(confusion_as_df, annot=True)
        save_dir = os.path.join('figures', name)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'{approach_name}_{dataset_name}_{eval_type}.png'))
        plt.savefig(os.path.join(save_dir, f'{approach_name}_{dataset_name}_{eval_type}.pdf'))
        plt.close()


def save_plot(name: str, dir_name: str):
    os.makedirs(dir_name, exist_ok=True)
    plt.savefig(os.path.join('figures', f'{name}.png'))
    plt.savefig(os.path.join('figures', f'{name}.pdf'))
    plt.close()


def plot_correct_spans(data: PlottingData, name: str):
    plt.figure()
    data_as_df = data.to_df()

    for eval_type, approach_name, dataset_name in zip(data.eval_types, data.approaches, data.datasets):
        data_to_plot = data_as_df[data_as_df['eval_type'] == eval_type]
        data_to_plot = data_to_plot[data_to_plot['approach'] == approach_name]
        data_to_plot = data_to_plot[data_to_plot['dataset'] == dataset_name]

        seaborn.barplot(x='approach', y='n_gold', data=data_to_plot, estimator=sum, color='red')
        seaborn.barplot(x='approach', y='n_ok', data=data_to_plot, estimator=sum, color='blue')

        save_dir = os.path.join('figures', name)
        save_name = f'{approach_name}_{dataset_name}_{eval_type}'

        save_plot(save_name, save_dir)


def plot_f1_scores(data: PlottingData, y_label: str, name: str, hue: str, markers: typing.List[str],
                   dataset: str = None, eval_type: str = None, approach: str = None, f1_type: str = None):
    data = data.to_df()
    if dataset is not None:
        data = data[data['dataset'] == dataset]
    if eval_type is not None:
        data = data[data['eval_type'] == eval_type]
    if approach is not None:
        data = data[data['approach'] == approach]
    if f1_type is not None:
        data = data[data['f1-type'] == f1_type]

    plt.figure()

    if f1_type is None:
        for i, _f1_type in enumerate(f1_types):
            ax = sns.stripplot(x='approach', y='f1', hue=hue, marker=markers[i],
                               size=7.5, jitter=.225,
                               data=data[data['f1-type'] == _f1_type])

        handles = []
        palette = sns.color_palette()

        for i, dataset in enumerate(data['dataset'].unique()):
            handles.append(lines.Line2D(
                [0], [0],
                marker='o', color='w',
                label=f'{dataset}',
                linestyle='',
                markerfacecolor=palette[i], markersize=9))

        dataset_legend = plt.legend(handles=handles,
                                    loc='upper right', bbox_to_anchor=(0.5, 1.05),
                                    ncol=2, prop={'size': 10}, markerscale=.95,
                                    borderpad=.15, labelspacing=.05, handletextpad=0.05)

        handles = []
        for i, _f1_type in enumerate(f1_types.values()):
            handles.append(lines.Line2D(
                [0], [0],
                marker=markers[i], color='black',
                label=f'{_f1_type}',
                linestyle='',
                fillstyle='none',
                markerfacecolor=palette[0], markersize=9))

        metric_legend = plt.legend(handles=handles,
                                   loc='upper left', bbox_to_anchor=(0.5, 1.05),
                                   ncol=2, prop={'size': 10}, markerscale=.95,
                                   borderpad=.15, labelspacing=.05, handletextpad=0.05)

        ax.add_artist(dataset_legend)
        ax.add_artist(metric_legend)
    else:
        sns.catplot(x='approach', y='f1', hue=hue, kind='swarm', data=data, legend=False)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                   ncol=4, prop={'size': 8}, markerscale=.75)

    plt.ylim(-.05, 1.05)
    plt.yticks(fontsize=14)
    plt.xticks(rotation=90, fontsize=14)
    plt.xlabel('')
    plt.ylabel(y_label, fontsize=14)

    plt.tight_layout()

    save_plot(name, 'figures')


def plot_binned_f1_scores(data: BinnedPlottingData, y_label: str, name: str, hue: str, markers: typing.List[str],
                          dataset: str = None, eval_type: str = None, approach: str = None, f1_type: str = None):
    data = data.to_df()
    if dataset is not None:
        data = data[data['dataset'] == dataset]
    if eval_type is not None:
        data = data[data['eval_type'] == eval_type]
    if approach is not None:
        data = data[data['approach'] == approach]
    if f1_type is not None:
        data = data[data['f1-type'] == f1_type]

    plt.figure()
    seaborn.lineplot(x='bin_value', y='f1', hue='approach', data=data)
    seaborn.scatterplot(x='bin_value', y='f1', hue='approach', size='bin_count', data=data, legend=False)

    save_plot(name, 'figures')


def plot_f1_scores_by_relation_variability(data: VariabilityPlottingData, y_label: str, name: str,
                                           markers: typing.List[str], eval_type: str = None,
                                           approach: str = None, f1_type: str = None):
    data = data.to_df()
    if eval_type is not None:
        data = data[data['eval_type'] == eval_type]
    if approach is not None:
        data = data[data['approach'] == approach]
    if f1_type is not None:
        data = data[data['f1-type'] == f1_type]

    plt.figure()
    seaborn.lmplot(x='variability', y='f1', hue='approach', data=data)

    save_plot(f'{name}_all', 'figures')
    plt.close()

    for approach in data['approach'].unique():
        data_by_approach = data[data['approach'] == approach]

        plt.figure()
        r, p = stats.pearsonr(x=data_by_approach['variability'], y=data_by_approach['f1'])
        regplot = seaborn.jointplot(x='variability', y='f1', data=data_by_approach, kind='reg')
        phantom, = regplot.ax_joint.plot([], [], linestyle='', alpha=0)
        regplot.ax_joint.legend([phantom], [f'r={r:.4f}, p={p:.4f}'])
        #seaborn.regplot(x='variability', y='f1', data=data_by_approach)


        save_plot(f'{name}_{approach}', 'figures')
        plt.close()


def get_parsed(dataset_result: DatasetResult,
               result_parser: parser.BaseParser,
               original_test_data: model.DataSet,
               original_valid_data: model.DataSet) -> typing.Tuple[typing.Iterable[metrics.LabeledSample],
                                                                   typing.Iterable[metrics.LabeledSample]]:
    test_parsed = list(result_parser.parse(dataset_result.test_results, original_test_data))
    valid_parsed = list(result_parser.parse(dataset_result.valid_results, original_valid_data))
    parser.BaseParser.insert_missing_samples(test_parsed, original_test_data)
    parser.BaseParser.insert_missing_samples(valid_parsed, original_valid_data)
    return test_parsed, valid_parsed


def get_label_set(dataset_result: typing.Iterable[metrics.LabeledSample]) -> typing.Set[str]:
    label_set: typing.Set[str] = set()
    for sample in dataset_result:
        for true in sample.labels:
            label_set.add(true.tag)
        for pred in sample.prediction:
            label_set.add(pred.tag)
    return label_set


@dataclasses.dataclass
class RelationPredictionFeatures:
    sample_id: str
    prediction_correct: bool

    sentence_length: int
    num_relations: int
    token_distance: int
    char_distance: int

    variability: float

    @staticmethod
    def header():
        return ['sample id', 'correct',
                'sentence length', 'num relations',
                'token distance', 'char distance',
                'variability']

    def to_csv_row(self):
        return [self.sample_id, self.prediction_correct,
                self.sentence_length, self.num_relations,
                self.token_distance, self.char_distance,
                self.variability]


def get_variability(original_dataset: model.DataSet) -> typing.Dict[str, float]:
    token_count_by_relation: typing.Dict[str, typing.Dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int))
    for sample in original_dataset.samples:
        for relation in sample.relations:
            head_token_start_index = min(relation.head.token_indices)
            head_token = sample.tokens[head_token_start_index]
            token_count_by_relation[relation.type][head_token.text] += 1

            tail_token_start_index = min(relation.tail.token_indices)
            tail_token = sample.tokens[tail_token_start_index]
            token_count_by_relation[relation.type][tail_token.text] += 1

    ret: typing.Dict[str, float] = {}
    for relation_type, counts in token_count_by_relation.items():
        num_total_tokens = sum(counts.values())
        num_unique_tokens = len(counts.keys())
        min_variability = 1.0 / num_total_tokens

        # in range min_variability;1.0
        unscaled_variability = num_unique_tokens / num_total_tokens

        # scale to range 0.0;1.0
        variability = (1.0 / (1.0 - min_variability)) * (unscaled_variability - min_variability)

        assert variability <= 1.0

        ret[relation_type] = variability
    return ret


def is_correct_prediction(relation: model.Relation, original_sample: model.Sample,
                          matches: typing.List[typing.Tuple[parser.LabeledRelation, parser.LabeledRelation]]) -> bool:
    for gold, _ in matches:
        head_token_start_index = min(relation.head.token_indices)
        head_token = original_sample.tokens[head_token_start_index]
        head_token_start_char = head_token.start_char_index

        tail_token_start_index = min(relation.tail.token_indices)
        tail_token = original_sample.tokens[tail_token_start_index]
        tail_token_start_char = tail_token.start_char_index

        token_index_matches = gold.head.start == head_token_start_index and gold.tail.start == tail_token_start_index
        char_index_matches = gold.head.start == head_token_start_char and gold.tail.start == tail_token_start_char

        if token_index_matches or char_index_matches:
            return True

    return False


def get_prediction_features(dataset_result: typing.Iterable[metrics.LabeledSample], original_dataset: model.DataSet,
                            span_matcher: metrics.BaseMatcher = None) -> typing.List[RelationPredictionFeatures]:
    if span_matcher is None:
        span_matcher = metrics.BoundaryRelationMatcher()

    variability_by_relation_type = get_variability(original_dataset)

    prediction_features: typing.List[RelationPredictionFeatures] = []

    id_to_sample: typing.Dict[str, model.Sample] = {str(s.id): s for s in original_dataset.samples}
    for labeled_sample in dataset_result:
        match_result = span_matcher.match(labeled_sample)

        original_sample = id_to_sample[str(labeled_sample.sample_id)]
        sentence_length = len(original_sample.tokens)
        num_relations = len(original_sample.relations)
        for relation in original_sample.relations:
            head_token_start_index = min(relation.head.token_indices)
            head_token = original_sample.tokens[head_token_start_index]

            tail_token_start_index = min(relation.tail.token_indices)
            tail_token = original_sample.tokens[tail_token_start_index]

            distance_in_tokens = abs(tail_token_start_index - head_token_start_index)
            distance_in_chars = abs(tail_token.start_char_index - head_token.start_char_index)

            variability = variability_by_relation_type[relation.type]

            is_correct = is_correct_prediction(relation, original_sample, match_result.matches)

            prediction_features.append(RelationPredictionFeatures(sample_id=original_sample.id,
                                                                  prediction_correct=is_correct,
                                                                  sentence_length=sentence_length,
                                                                  num_relations=num_relations,
                                                                  token_distance=distance_in_tokens,
                                                                  char_distance=distance_in_chars,
                                                                  variability=variability))
    return prediction_features


def print_csv(data: typing.Dict):
    for approach_id, dataset_results in data.items():
        for dataset_id, dataset_result in dataset_results.items():
            features = get_prediction_features(dataset_result['test_results'], dataset_result['test_original'])

            csv_out_dir = os.path.join('figures', 'features')
            os.makedirs(csv_out_dir, exist_ok=True)

            csv_out_path = os.path.join(csv_out_dir, f'{approach_id}_{dataset_id}.csv')
            with open(csv_out_path, 'w', encoding='utf8', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(RelationPredictionFeatures.header())
                for feature in features:
                    csv_writer.writerow(feature.to_csv_row())


def generate_plotting_data(data: typing.Dict, f1_types: typing.Dict[str, str], matcher: metrics.BaseMatcher):
    plotting_data = PlottingData()
    for approach_id, dataset_results in data.items():
        for dataset_id, dataset_result in dataset_results.items():
            print(
                f'Generating plotting data for "{dataset_result["approach_name"]}" on "{dataset_result["dataset_name"]}"')
            for subset in ['test', 'valid']:
                results = dataset_result[f'{subset}_results']
                label_list = list(get_label_set(results))
                for f1_type in f1_types:
                    score = metrics.calculate_f1(results, matcher, f1_type)
                    confusion_matrix, n_gold_entities, _, n_ok_entities = metrics.calculate_confusion_matrix(
                        results, label_list,
                        metrics.BoundaryRelationMatcher()
                    )
                    plotting_data.add(approach=dataset_result['approach_name'],
                                      dataset=dataset_result['dataset_name'],
                                      f1_type=f1_type, eval_type=subset,
                                      confusion_matrix=confusion_matrix, labels=label_list,
                                      n_ok_entities=n_ok_entities, n_gold_entities=n_gold_entities,
                                      f1_score=score.f1, r_score=score.recall, p_score=score.precision)
    return plotting_data


def generate_binned_samples_plotting_data(data: typing.Dict[str, typing.Dict[str, typing.Dict]], f1_type: str,
                                          matcher: metrics.BaseMatcher,
                                          bin_predicate: typing.Callable[[binning.PredictionAndOriginal], float],
                                          num_bins: typing.Callable[[typing.Iterable[model.Sample]], int]
                                          ) -> BinnedPlottingData:
    plotting_data = BinnedPlottingData()
    for approach_id, dataset_results in data.items():
        for dataset_id, dataset_result in dataset_results.items():
            print(f'Generating binned plotting data for "{dataset_result["approach_name"]}" on "{dataset_result["dataset_name"]}"')
            for subset in ['test', 'valid']:
                results: typing.Iterable[metrics.LabeledSample] = dataset_result[f'{subset}_results']
                label_list = list(get_label_set(results))
                originals: typing.Iterable[model.Sample] = dataset_result[f'{subset}_original'].samples
                samples = zip(results, originals)
                binned_samples, bin_edges = binning.bin_predictions_by(samples, bin_predicate, num_bins(originals))
                for bin_id, samples in binned_samples.items():
                    results = [s[0] for s in samples]
                    score = metrics.calculate_f1(results, matcher, f1_type)
                    confusion_matrix, n_gold_entities, _, n_ok_entities = metrics.calculate_confusion_matrix(
                        results, label_list,
                        metrics.BoundaryRelationMatcher()
                    )
                    plotting_data.add(
                        approach=dataset_result['approach_name'],
                        dataset=dataset_result['dataset_name'],
                        f1_type=f1_type,
                        eval_type=subset,
                        confusion_matrix=confusion_matrix,
                        labels=label_list,
                        n_ok_entities=n_ok_entities,
                        n_gold_entities=n_gold_entities,
                        f1_score=score.f1,
                        r_score=score.recall,
                        p_score=score.precision,
                        bin_id=bin_id,
                        bin_value=bin_edges[bin_id],
                        bin_count=len(samples)
                    )
    return plotting_data


def generate_variability_plotting_data(data: typing.Dict[str, typing.Dict[str, typing.Dict]], f1_type: str,
                                       matcher: metrics.BaseMatcher) -> VariabilityPlottingData:
    plotting_data = VariabilityPlottingData()
    for approach_id, dataset_results in data.items():
        for dataset_id, dataset_result in dataset_results.items():
            print(f'Generating variability plotting data for "{dataset_result["approach_name"]}" on "{dataset_result["dataset_name"]}"')
            for subset in ['test', 'valid']:
                original = dataset_result[f'{subset}_original']
                variability_by_relation_type = get_variability(original)
                subset_results: typing.Iterable[metrics.LabeledSample] = dataset_result[f'{subset}_results']
                results_by_relation_type: typing.Dict[str, typing.List[metrics.LabeledSample]] = collections.defaultdict(list)
                for sample in subset_results:
                    if len(sample.labels) == 0:
                        continue
                    try:
                        max_variability_relation = max(sample.labels, key=lambda x: variability_by_relation_type[x.tag])
                    except KeyError as e:
                        raise KeyError(f'Unknown relation type "{e.args[0]}" in sample with id {sample.sample_id}, known relations are {list(variability_by_relation_type.keys())}')
                    results_by_relation_type[max_variability_relation.tag].append(sample)
                for relation_type, samples in results_by_relation_type.items():
                    score = metrics.calculate_f1(samples, matcher, f1_type)
                    variability = variability_by_relation_type[relation_type]
                    plotting_data.add(
                        approach=dataset_result['approach_name'],
                        dataset=dataset_result['dataset_name'],
                        f1_type=f1_type,
                        eval_type=subset,
                        confusion_matrix=np.array([]),
                        labels=[],
                        n_ok_entities=0,
                        n_gold_entities=0,
                        f1_score=score.f1,
                        r_score=score.recall,
                        p_score=score.precision,
                        variability=variability,
                        relation_type=relation_type
                    )
    return plotting_data


def parse_results(approach_results: typing.List[ApproachResult]):
    parsed_data: typing.Dict[str, typing.Dict[str, dict]] = collections.defaultdict(dict)

    for approach in approach_results:
        file_parser = metrics.available_result_parsers[approach.id]
        for result in approach.datasets:
            print(f'Parsing results of "{approach.name}" for dataset "{result.name}"')

            parsed_test_results, parsed_valid_results = get_parsed(result, file_parser,
                                                                   original_data[result.id]['test'],
                                                                   original_data[result.id]['valid'])

            parsed_data[approach.id][result.id] = {
                'approach_name': approach.name,
                'dataset_name': result.name,
                'result': result,
                'test_results': parsed_test_results,
                'valid_results': parsed_valid_results,
                'test_original': original_data[result.id]['test'],
                'valid_original': original_data[result.id]['valid']
            }

    return parsed_data


def get_token_distance(sample: model.Sample, op: typing.Callable[[typing.Iterable[float]], float],
                       mode='char') -> float:
    assert mode in ['char', 'token']
    if len(sample.relations) <= 0:
        raise ValueError(f'Cant get distance between entities if no relations exist')
    distances = []
    for r in sample.relations:
        head_token_start_index = min(r.head.token_indices)
        head_token = sample.tokens[head_token_start_index]

        tail_token_start_index = min(r.tail.token_indices)
        tail_token = sample.tokens[tail_token_start_index]

        if mode == 'char':
            distance = abs(tail_token.start_char_index - head_token.start_char_index)
        else:
            distance = abs(tail_token_start_index - head_token_start_index)

        distances.append(distance)
    return op(distances)


base_original_data_dir = os.path.join('data', 'export')

original_data = {
    'conll04': {
        'test': convert.available_importers['line-json']().load(
            [os.path.join(base_original_data_dir, 'conll04', 'jsonl', 'test.json')]),
        'valid': convert.available_importers['line-json']().load(
            [os.path.join(base_original_data_dir, 'conll04', 'jsonl', 'valid.json')]),
    },
    'fewrel': {
        'test': convert.available_importers['line-json']().load(
            [os.path.join(base_original_data_dir, 'fewrel', 'jsonl', 'test.json')]),
        'valid': convert.available_importers['line-json']().load(
            [os.path.join(base_original_data_dir, 'fewrel', 'jsonl', 'valid.json')]),
    },
    'nyt10': {
        'test': convert.available_importers['line-json']().load(
            [os.path.join(base_original_data_dir, 'nyt10', 'jsonl', 'test.json')]),
        'valid': convert.available_importers['line-json']().load(
            [os.path.join(base_original_data_dir, 'nyt10', 'jsonl', 'valid.json')]),
    },
    'semeval': {
        'test': convert.available_importers['line-json']().load(
            [os.path.join(base_original_data_dir, 'semeval', 'jsonl', 'test.json')]),
        'valid': convert.available_importers['line-json']().load(
            [os.path.join(base_original_data_dir, 'semeval', 'jsonl', 'valid.json')]),
    }
}

if __name__ == '__main__':
    f1_types = {'micro': 'micro', 'macro-relations': 'macro (rel)', 'macro-documents': 'macro (doc)'}
    default_markers = ['o', 'v', 's', 'D', 'P', '*']

    parsed_results = parse_results(results)

    relation_plotting_data = generate_plotting_data(parsed_results, f1_types, metrics.ExactRelationMatcher())
    for f1_type in f1_types:
        # cat plot on test data for this f1 type only
        plot_f1_scores(data=relation_plotting_data, hue='dataset',
                       name=f'catplot-test-{f1_type}', y_label='F1 score',
                       f1_type=f1_type, eval_type='test', markers=default_markers)

        # cat plot on valid data for this f1 type only
        plot_f1_scores(data=relation_plotting_data, hue='dataset',
                       name=f'catplot-valid-{f1_type}', y_label='F1 score',
                       f1_type=f1_type, eval_type='valid', markers=default_markers)

    # cat plot on valid data for all f1 types
    plot_f1_scores(data=relation_plotting_data, hue='dataset',
                   name=f'catplot-valid', y_label='F1 score',
                   eval_type='valid', markers=default_markers)

    # cat plot on test data for all f1 types
    plot_f1_scores(data=relation_plotting_data, hue='dataset',
                   name=f'catplot-test', y_label='F1 score',
                   eval_type='test', markers=default_markers)

    # cat plot on both data subsets for all f1 types
    plot_f1_scores(data=relation_plotting_data, hue='dataset',
                   name=f'catplot-combined', y_label='F1 score', markers=default_markers)

    entity_plotting_data = generate_plotting_data(parsed_results, f1_types, metrics.EntityMatcher())
    for f1_type in f1_types:
        # cat plot on test data for this f1 type only
        plot_f1_scores(data=entity_plotting_data, hue='dataset',
                       name=f'entity-catplot-test-{f1_type}', y_label='F1 score',
                       f1_type=f1_type, eval_type='test', markers=default_markers)

        # cat plot on valid data for this f1 type only
        plot_f1_scores(data=entity_plotting_data, hue='dataset',
                       name=f'entity-catplot-valid-{f1_type}', y_label='F1 score',
                       f1_type=f1_type, eval_type='valid', markers=default_markers)

    # cat plot on valid data for all f1 types
    plot_f1_scores(data=entity_plotting_data, hue='dataset',
                   name=f'entity-catplot-valid', y_label='F1 score',
                   eval_type='valid', markers=default_markers)

    # cat plot on test data for all f1 types
    plot_f1_scores(data=entity_plotting_data, hue='dataset',
                   name=f'entity-catplot-test', y_label='F1 score',
                   eval_type='test', markers=default_markers)

    # cat plot on both data subsets for all f1 types
    plot_f1_scores(data=entity_plotting_data, hue='dataset',
                   name=f'entity-catplot-combined', y_label='F1 score', markers=default_markers)



    binned_plotting_data = generate_binned_samples_plotting_data(parsed_results,
                                                                 'micro',
                                                                 metrics.ExactRelationMatcher(),
                                                                 lambda x: len(x[1].tokens),
                                                                 num_bins=lambda _: 5)
    plot_binned_f1_scores(data=binned_plotting_data, y_label='F1 Score', name=f'binned_length_fewrel',
                          hue='approach', markers=default_markers, eval_type='test', dataset='FewRel')
    plot_binned_f1_scores(data=binned_plotting_data, y_label='F1 Score', name=f'binned_length_conll04',
                          hue='approach', markers=default_markers, eval_type='test', dataset='ConLL 04')
    plot_binned_f1_scores(data=binned_plotting_data, y_label='F1 Score', name=f'binned_length_nyt10',
                          hue='approach', markers=default_markers, eval_type='test', dataset='NYT10')
    plot_binned_f1_scores(data=binned_plotting_data, y_label='F1 Score', name=f'binned_length_semeval',
                          hue='approach', markers=default_markers, eval_type='test', dataset='Semeval')

    binned_plotting_data = generate_binned_samples_plotting_data(parsed_results,
                                                                 'micro',
                                                                 metrics.ExactRelationMatcher(),
                                                                 lambda x: len(x[1].relations),
                                                                 num_bins=lambda original_samples: max(
                                                                     [len(x.relations) for x in original_samples]))
    plot_binned_f1_scores(data=binned_plotting_data, y_label='F1 Score', name=f'binned_relations_fewrel',
                          hue='approach', markers=default_markers, eval_type='test', dataset='FewRel')
    plot_binned_f1_scores(data=binned_plotting_data, y_label='F1 Score', name=f'binned_relations_conll04',
                          hue='approach', markers=default_markers, eval_type='test', dataset='ConLL 04')
    plot_binned_f1_scores(data=binned_plotting_data, y_label='F1 Score', name=f'binned_relations_nyt10',
                          hue='approach', markers=default_markers, eval_type='test', dataset='NYT10')
    plot_binned_f1_scores(data=binned_plotting_data, y_label='F1 Score', name=f'binned_relations_semeval',
                          hue='approach', markers=default_markers, eval_type='test', dataset='Semeval')

    binned_plotting_data = generate_binned_samples_plotting_data(parsed_results,
                                                                 'micro',
                                                                 metrics.ExactRelationMatcher(),
                                                                 lambda x: get_token_distance(x[1], max),
                                                                 num_bins=lambda _: 5)
    plot_binned_f1_scores(data=binned_plotting_data, y_label='F1 Score', name=f'binned_distance_fewrel',
                          hue='approach', markers=default_markers, eval_type='test', dataset='FewRel')
    plot_binned_f1_scores(data=binned_plotting_data, y_label='F1 Score', name=f'binned_distance_conll04',
                          hue='approach', markers=default_markers, eval_type='test', dataset='ConLL 04')
    plot_binned_f1_scores(data=binned_plotting_data, y_label='F1 Score', name=f'binned_distance_nyt10',
                          hue='approach', markers=default_markers, eval_type='test', dataset='NYT10')
    plot_binned_f1_scores(data=binned_plotting_data, y_label='F1 Score', name=f'binned_distance_semeval',
                          hue='approach', markers=default_markers, eval_type='test', dataset='Semeval')


    variability_plotting_data = generate_variability_plotting_data(parsed_results, 'micro',
                                                                   metrics.ExactRelationMatcher())

    plot_f1_scores_by_relation_variability(data=variability_plotting_data, y_label='F1 Score', name=f'variability',
                                           markers=default_markers, eval_type='test')

    """
    # correct vs expected predictions
    plot_correct_spans(data=plotting_data, name='gold_vs_ok')
    
    # confusion matrices
    plot_confusion_matrix(data=plotting_data, name='confusion')
    """

    """
    # csv prediction data
    print_csv(parsed_results)
    """

    # tables
    write_result_tables(relation_plotting_data)
