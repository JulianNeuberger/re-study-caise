import json
import os
import typing

import model
from convert.exporter import base


class DualDecExporter(base.BaseExporter):
    def _dump_sample_to_dict(self, sample: model.Sample) -> typing.Dict:
        return {
            'tokens': [t.text for t in sample.tokens],
            'spo_list': [(r.head.text, r.type, r.tail.text) for r in sample.relations],
            'spo_details': [
                (
                    min(r.head.token_indices), max(r.head.token_indices) + 1, r.head.ner_tag,
                    r.type,
                    min(r.tail.token_indices), max(r.tail.token_indices) + 1, r.tail.ner_tag
                )
                for r in sample.relations
            ],
            'pos_tags': [t.pos_tag for t in sample.tokens]
        }

    def save(self, data_set: model.DataSet, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        serialized_samples = [self._dump_sample_to_dict(s) for s in data_set.samples]
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(serialized_samples, f)
