import json
import os
import typing

import model
from convert.exporter import base


class SpertExporter(base.BaseExporter):
    def _dump_entity_as_dict(self, entity: model.Entity) -> typing.Dict:
        return {
            'type': entity.ner_tag,
            'start': min(entity.token_indices),
            'end': max(entity.token_indices) + 1
        }

    def _dump_relation_as_dict(self, entities: typing.List[model.Entity],
                               relation: model.Relation) -> typing.Dict:
        head_idx = entities.index(relation.head)
        tail_idx = entities.index(relation.tail)
        return {
            'type': relation.type,
            'head': head_idx,
            'tail': tail_idx
        }

    def _dump_sample_as_dict(self, sample: model.Sample) -> typing.Dict:
        return {
            'tokens': [t.text for t in sample.tokens],
            'entities': [self._dump_entity_as_dict(e) for e in sample.entities],
            'relations': [self._dump_relation_as_dict(sample.entities, r) for r in sample.relations],
            'orig_id': sample.id
        }

    def save(self, data_set: model.DataSet, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        samples_as_dicts = [self._dump_sample_as_dict(s) for s in data_set.samples]
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(samples_as_dicts, f)
