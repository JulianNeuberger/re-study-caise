import typing

import model
from analyze import base
from analyze.base import AnalysisResult


class PosTagsAnalysisStep(base.AnalysisStep):
    @staticmethod
    def get_pos_tags(sample: model.Sample, entity: model.Entity, before: int, after: int) -> typing.Optional[typing.List[str]]:
        indices = entity.token_indices
        context_before = [i for i in range(min(indices) - before, min(indices)) if i >= 0]
        context_after = [i for i in range(max(indices), max(indices) + after) if i < len(sample.tokens)]

        if len(context_after) != after:
            return None
        if len(context_before) != before:
            return None

        assert all([i not in indices for i in context_before])
        assert all([i not in indices for i in context_after])

        indices = context_before + indices + context_after
        return [sample.tokens[i].pos_tag for i in indices]

    @staticmethod
    def _add(nested_dict: typing.Dict[typing.Any, typing.Dict[typing.Any, float]], k1: typing.Any, k2: typing.Any,
             count: float = 1):
        if k1 not in nested_dict:
            nested_dict[k1] = {}

        if k2 not in nested_dict[k1]:
            nested_dict[k1][k2] = 0

        nested_dict[k1][k2] += count

    def calculate_results(self, dataset: model.DataSet) -> typing.List[AnalysisResult]:
        ner_tag_to_pos_tags: typing.Dict[str, typing.Dict[typing.Tuple, int]] = {}
        ner_tag_to_pos_tags_w_context: typing.Dict[str, typing.Dict[typing.Tuple, int]] = {}
        differences: typing.Dict[str, typing.Dict[typing.Tuple, int]] = {}

        for s in dataset.samples:
            for e in s.entities:
                pos_tags_wo_context = self.get_pos_tags(s, e, 0, 0)
                assert pos_tags_wo_context is not None
                self._add(ner_tag_to_pos_tags, e.ner_tag, tuple(pos_tags_wo_context))
                pos_tags_w_context = self.get_pos_tags(s, e, 1, 0)
                if pos_tags_w_context:
                    self._add(ner_tag_to_pos_tags_w_context, e.ner_tag, tuple(pos_tags_w_context))

        for ner_tag in ner_tag_to_pos_tags:
            for pos_tags in ner_tag_to_pos_tags[ner_tag]:
                if pos_tags not in ner_tag_to_pos_tags_w_context[ner_tag]:
                    # this was never labeled differently
                    continue
                num_labeled_without_context = ner_tag_to_pos_tags[ner_tag][pos_tags]
                num_labeled_with_context = ner_tag_to_pos_tags_w_context[ner_tag][pos_tags]
                if num_labeled_with_context + num_labeled_without_context < 10.0:
                    continue
                smaller = min(num_labeled_without_context, num_labeled_with_context)
                bigger = max(num_labeled_without_context, num_labeled_with_context)
                self._add(differences, ner_tag, f'[{pos_tags[0]}] {" ".join(pos_tags[1:])}', smaller / bigger)

        return [
            base.AnalysisResult('entity_pos_tags', ner_tag_to_pos_tags,
                                'Number of entities with a specific pos tag combination'),
            base.AnalysisResult('entity_pos_tags_w_context', ner_tag_to_pos_tags_w_context,
                                'Number of entities with a specific pos tag combination, '
                                'using the immediate predecessor token as context'),
            base.AnalysisResult('different_tagging', differences,
                                'Number of tagging instances where context was ignored')
        ]
