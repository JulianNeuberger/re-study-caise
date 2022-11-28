import json
import os
import typing

import model
from convert.exporter import base


class DocRedExporter(base.BaseExporter):
    @staticmethod
    def _dump_entity_as_dict(entity: model.Entity,
                             entity_to_id: typing.Dict[model.Entity, int]) -> typing.Dict:
        entity_to_id[entity] = len(entity_to_id)
        return {
            'pos': (min(entity.token_indices), max(entity.token_indices) + 1),
            'type': entity.ner_tag,
            'sent_id': 0,
            'name': entity.text
        }

    @staticmethod
    def _dump_relation_as_dict(relation: model.Relation,
                               sample: model.Sample) -> typing.Dict:
        return {
            'r': relation.type,
            'h': sample.entities.index(relation.head),
            't': sample.entities.index(relation.tail),
            'evidence': [0]
        }

    def _dump_sample_as_dict(self, sample: model.Sample) -> typing.Dict:
        entity_to_id = {}
        serialized_entities = [self._dump_entity_as_dict(e, entity_to_id) for e in sample.entities]

        return {
            'vertexSet': [[s] for s in serialized_entities],
            'labels': [self._dump_relation_as_dict(r, sample) for r in sample.relations],
            'title': '',
            'sents': [[t.text for t in sample.tokens]]
        }

    def save(self, data_set: model.DataSet, file_path: str) -> None:
        serialized_samples = []
        for s in data_set.samples:
            serialized_samples.append(self._dump_sample_as_dict(s))

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(serialized_samples, f)
