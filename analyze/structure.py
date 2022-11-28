import typing

import numpy as np

import model
from analyze import base
from analyze.base import AnalysisResult


class NumSamplesAnalysisStep(base.AnalysisStep):
    def calculate_results(self, dataset: model.DataSet) -> typing.List[base.AnalysisResult]:
        return [
            base.AnalysisResult('num_samples', len(dataset.samples), 'number of samples')
        ]


class RelationTypesAnalysisStep(base.AnalysisStep):
    def __init__(self, return_relation_count: typing.List[str] = None):
        if return_relation_count is None:
            return_relation_count = []
        assert all([r in ['all', 'min', 'max'] for r in return_relation_count])
        self._return_relation_count = return_relation_count

    def calculate_results(self, dataset: model.DataSet) -> typing.List[base.AnalysisResult]:
        relation_types: typing.Dict[str, int] = {}
        for s in dataset.samples:
            for r in s.relations:
                if r.type not in relation_types:
                    relation_types[r.type] = 0
                relation_types[r.type] += 1
        ret = [
            base.AnalysisResult('num_relations', sum(relation_types.values()), 'total number of relation instances'),
            base.AnalysisResult('num_relation_types', len(relation_types), 'number of unique relation type'),
        ]

        if 'all' in self._return_relation_count:
            ret += [base.AnalysisResult('relation_types', relation_types, 'relation types')]
        if 'min' in self._return_relation_count:
            ret += [base.AnalysisResult('min_relation_type',
                                        min(relation_types.items(), key=lambda i: i[1]), 'least common relation')]
        if 'max' in self._return_relation_count:
            ret += [base.AnalysisResult('max_relation_type',
                                        max(relation_types.items(), key=lambda i: i[1]), 'most common relation')]

        return ret


class VocabSizeAnalysisStep(base.AnalysisStep):
    def __init__(self, quantile: typing.Optional[float] = None):
        if quantile is not None:
            assert 0.0 <= quantile <= 1.0
        self._quantile = quantile

    def calculate_results(self, dataset: model.DataSet) -> typing.List[base.AnalysisResult]:
        vocab = {}
        num_tokens = 0
        for s in dataset.samples:
            for t in s.tokens:
                num_tokens += 1
                if t.text not in vocab:
                    vocab[t.text] = 0
                vocab[t.text] += 1
        # sort vocab desc by token occurrences
        vocab = {
            k: v
            for k, v
            in sorted(vocab.items(), key=lambda i: i[1], reverse=True)
        }

        ret = [
            base.AnalysisResult('vocab_size', len(vocab), 'vocabulary size'),
            base.AnalysisResult('num_tokens', num_tokens, 'number of tokens in entire dataset')
        ]
        if self._quantile is not None:
            total = sum(vocab.values())
            cur = 0
            cutoff = int(total * self._quantile)
            num_tokens_in_quantile = 0
            for token in vocab:
                cur += vocab[token]
                num_tokens_in_quantile += 1
                if cur >= cutoff:
                    break
            ret += [base.AnalysisResult(f'tokens_in_{self._quantile:.0%}_quantile', num_tokens_in_quantile,
                                        f'number of tokens in {self._quantile:.0%} quantile')]
        return ret


class NerTagAnalysisStep(base.AnalysisStep):
    def calculate_results(self, dataset: model.DataSet) -> typing.List[base.AnalysisResult]:
        ner_tags = set()
        for s in dataset.samples:
            for e in s.entities:
                ner_tags.add(e.ner_tag)
        return [
            base.AnalysisResult('num_ner_tags', len(ner_tags), 'number of unique NER tags')
        ]


class SampleLengthAnalysisStep(base.AnalysisStep):
    def calculate_results(self, dataset: model.DataSet) -> typing.List[base.AnalysisResult]:
        lengths = np.array([len(s.tokens) for s in dataset.samples])
        return [
            base.AnalysisResult('min_sample_length', np.min(lengths), 'least number of tokens per sample'),
            base.AnalysisResult('max_sample_length', np.max(lengths), 'most number of tokens per sample'),
            base.AnalysisResult('average_sample_length', np.mean(lengths), 'average number of tokens per sample'),
        ]


class NumEntityAnalysisStep(base.AnalysisStep):
    def calculate_results(self, dataset: model.DataSet) -> typing.List[AnalysisResult]:
        num_entities = 0
        entity_tags = set()
        for s in dataset.samples:
            for e in s.entities:
                num_entities += 1
                entity_tags.add(e.ner_tag)
        return [
            base.AnalysisResult('num_entities', num_entities, 'number of entities'),
            base.AnalysisResult('num_entity_tags', len(entity_tags), 'size of the ner tag-set')
        ]


class RelationDistanceAnalysisStep(base.AnalysisStep):
    def calculate_results(self, dataset: model.DataSet) -> typing.List[AnalysisResult]:
        distances: typing.List[int] = []
        for s in dataset.samples:
            for r in s.relations:
                distance = abs(r.tail.token_indices[0] - r.head.token_indices[0])
                distances.append(distance)

        min_distance = 0
        max_distance = 0
        avg_distance = 0
        if len(distances) != 0:
            min_distance = min(distances)
            max_distance = max(distances)
            avg_distance = sum(distances) / len(distances)
        return [
            base.AnalysisResult('min_relation_distance', min_distance, 'minimal distance between relation heads'),
            base.AnalysisResult('max_relation_distance', max_distance, 'maximum distance between relation heads'),
            base.AnalysisResult('avg_relation_distance', avg_distance, 'average distance between relation heads')
        ]


class NegativeExamplesAnalysisStep(base.AnalysisStep):
    def calculate_results(self, dataset: model.DataSet) -> typing.List[AnalysisResult]:
        num_negative = 0
        num_positive = 0
        for s in dataset.samples:
            if len(s.relations) == 0:
                num_negative += 1
            else:
                num_positive += 1
        return [
            base.AnalysisResult('negative_examples', num_negative, 'samples with at least one relation'),
            base.AnalysisResult('positive_examples', num_positive, 'samples with exactly zero relations')
        ]


class EntityLengthAnalysisStep(base.AnalysisStep):
    def calculate_results(self, dataset: model.DataSet) -> typing.List[AnalysisResult]:
        entity_lengths: typing.Dict[int, int] = {}
        for s in dataset.samples:
            for e in s.entities:
                entity_length = len(e.token_indices)
                if entity_length not in entity_lengths:
                    entity_lengths[entity_length] = 0
                entity_lengths[entity_length] += 1
        return [
            base.AnalysisResult('min_entity_length', min(entity_lengths.keys()), 'shortest entity (in tokens)'),
            base.AnalysisResult('max_entity_length', max(entity_lengths.keys()), 'longest entity (in tokens)'),
            base.AnalysisResult(
                'avg_entity_length',
                sum([length * num for length, num in entity_lengths.items()]) / sum(entity_lengths.values()),
                'average entity length (in tokens)'
            ),
        ]


class RelationEntityTagAnalysisStep(base.AnalysisStep):
    def calculate_results(self, dataset: model.DataSet) -> typing.List[AnalysisResult]:
        entity_types: typing.Dict[str, typing.Dict[typing.Tuple[str, str], int]] = {}
        for s in dataset.samples:
            for r in s.relations:
                if r.type not in entity_types:
                    entity_types[r.type] = {}
                tags = (r.head.ner_tag, r.tail.ner_tag)
                if tags not in entity_types[r.type]:
                    entity_types[r.type][tags] = 0
                entity_types[r.type][tags] += 1
        return [
            base.AnalysisResult('relation_entity_tags', entity_types,
                                'co occurrence information for ner tags of head and tail entities in relations')
        ]


class NumRelationsPerSampleAnalysisStep(base.AnalysisStep):
    def calculate_results(self, dataset: model.DataSet) -> typing.List[AnalysisResult]:
        num_relations_per_sample: typing.Dict[int, int] = {}
        for s in dataset.samples:
            num_relations = len(s.relations)
            if num_relations not in num_relations_per_sample:
                num_relations_per_sample[num_relations] = 0
            num_relations_per_sample[num_relations] += 1
        num_relations_per_sample = {k: v for k, v in sorted(num_relations_per_sample.items(), key=lambda i: i[0])}
        return [
            base.AnalysisResult('num_relations_per_sample', num_relations_per_sample, 'number of relations per sample')
        ]
