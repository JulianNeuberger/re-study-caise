import json
import os
import typing

import model
from convert.exporter import base


class JointERExporter(base.BaseExporter):
    def calculate_ner_tag_for_entity(self, sample: model.Sample, entity: model.Entity) -> str:
        tokens = [sample.tokens[i] for i in entity.token_indices]
        # FIXME: map ner tags properly, if we have e.g. something like a BIO tag-set
        return tokens[0].ner_tag

    def _dump_sample(self, sample: model.Sample) -> typing.Dict:
        return {
            'id': sample.id,
            'tokens': [t.text for t in sample.tokens],
            'spo_list': [(r.head.text, r.type, r.tail.text) for r in sample.relations],
            'spo_details': [(min(r.head.token_indices), max(r.head.token_indices) + 1,
                             self.calculate_ner_tag_for_entity(sample, r.head),
                             r.type,
                             min(r.tail.token_indices), max(r.tail.token_indices) + 1,
                             self.calculate_ner_tag_for_entity(sample, r.tail)) for r in sample.relations],
            'pos_tags': [t.pos_tag for t in sample.tokens]
        }

    def save(self, data_set: model.DataSet, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf8') as out:
            json.dump([self._dump_sample(s) for s in data_set.samples], out)
