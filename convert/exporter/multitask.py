import json
import os
import typing

import model
from convert import util
from convert.exporter import base


class ProgressiveMultiTaskExporter(base.BaseExporter):
    def _get_entity_tokens(self, sample: model.Sample, entity: model.Entity) -> typing.List[model.Token]:
        return [sample.tokens[i] for i in entity.token_indices]

    def _get_entity_ner_tag(self, sample: model.Sample, entity: model.Entity) -> str:
        return entity.ner_tag

    def _dump_relation_as_dict(self, sample: model.Sample, relation: model.Relation) -> typing.Dict:
        return {
            'em1Text': [t.text for t in self._get_entity_tokens(sample, relation.head)],
            'em2Text': [t.text for t in self._get_entity_tokens(sample, relation.tail)],
            'label': relation.type,
            'em1Label': self._get_entity_ner_tag(sample, relation.head),
            'em2Label': self._get_entity_ner_tag(sample, relation.tail)
        }

    def _dump_sample_as_dict(self, sample: model.Sample) -> typing.Dict:
        return {
            'sentText': [t.text for t in sample.tokens],
            'relationMentions': [self._dump_relation_as_dict(sample, r) for r in sample.relations],
            'pos': [t.pos_tag for t in sample.tokens],
            'en_list': [
                [t.text for t in self._get_entity_tokens(sample, e)] for e in sample.entities
            ],
            'stan_ner': [util.plain_ner_tag_from_bioes(t.ner_tag) for t in sample.tokens]
        }

    def save(self, data_set: model.DataSet, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf8') as f:
            for sample in data_set.samples:
                sample_as_json = json.dumps(self._dump_sample_as_dict(sample))
                f.write(f'{sample_as_json}\n')
