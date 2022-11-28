import json
import os
import typing

import model
from convert.exporter import base


class CasRelExporter(base.BaseExporter):
    def _clean_text(self, text: str) -> str:
        text = text.lstrip('\"')
        text = text.strip('\r\n')
        text = text.rstrip('\"')
        text = text.replace(',', ' ,')
        text = text.replace("'s", " 's")
        text = text.lstrip()
        text = text.rstrip()
        text = text.rstrip('.')
        return text

    def _dump_relation_as_tuple(self, sample: model.Sample, relation: model.Relation) -> typing.Tuple:
        em1_text = ' '.join(sample.tokens[i].text for i in relation.head.token_indices)
        em2_text = ' '.join(sample.tokens[i].text for i in relation.tail.token_indices)
        return em1_text, relation.type, em2_text

    def _dump_sample_as_dict(self, sample: model.Sample) -> typing.Dict:
        sample_text = ' '.join([t.text for t in sample.tokens])
        return {
            'sentId': str(sample.id),
            'text': sample_text,
            'triple_list': [self._dump_relation_as_tuple(sample, r) for r in sample.relations]
        }

    def save(self, data_set: model.DataSet, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf8') as f:
            json.dump([self._dump_sample_as_dict(s) for s in data_set.samples], f)
