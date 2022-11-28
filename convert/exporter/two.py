import json
import os

from convert.exporter import base
import model


def _dump_sample_as_dict(sample: model.Sample):
    return {
        'id': sample.id,
        'tokens': [t.text for t in sample.tokens],
        'entities': [
            [
                min(e.token_indices),
                max(e.token_indices) + 1,
                e.ner_tag
            ] for e in sample.entities],
        'relations': [
            [
                min(r.head.token_indices),
                max(r.head.token_indices) + 1,
                min(r.tail.token_indices),
                max(r.tail.token_indices) + 1,
                r.type
            ]
            for r in sample.relations
        ]
    }


class TwoAreBetterThanOneExporter(base.BaseExporter):
    def save(self, data_set: model.DataSet, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        converted_samples = [_dump_sample_as_dict(sample) for sample in data_set.samples]
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump(converted_samples, f)
