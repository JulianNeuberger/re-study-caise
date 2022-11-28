import json
import os
import typing

import model
from convert.exporter import base


class RSANExporter(base.BaseExporter):
    def _get_entity_text(self, sample: model.Sample, entity: model.Entity) -> str:
        return ' '.join([t.text for i, t in enumerate(sample.tokens) if i in entity.token_indices])

    def _dump_sample(self, sample: model.Sample) -> typing.Dict:
        return {
            'id': str(sample.id),
            'sentText': ' '.join([t.text for t in sample.tokens]),
            'articleId': 'None',
            'relationMentions': [
                {
                    'em1Text': self._get_entity_text(sample, r.head),
                    'em2Text': self._get_entity_text(sample, r.tail),
                    'label': r.type
                }
                for r in sample.relations
            ],
            'entityMentions': [
                {
                    'start': min(e.token_indices),
                    'label': e.ner_tag,
                    'text': self._get_entity_text(sample, e)
                }
                for e in sample.entities
            ]
        }

    def save(self, data_set: model.DataSet, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf8') as f:
            for sample in data_set.samples:
                serialized_sample = json.dumps(self._dump_sample(sample))
                line = f'{serialized_sample}\n'
                f.write(line)
