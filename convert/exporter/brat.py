import os
import shutil
import typing

import model
from convert.exporter import base


class BratExporter(base.BaseExporter):
    def __init__(self, num_sentences_per_doc: int = 100):
        self._num_sents_per_doc = num_sentences_per_doc

    def _dumps_entity(self, sample: model.Sample, entity: model.Entity, entity_id: int,
                      total_text: str, text_offset: int) -> str:
        actual_last_index = entity.token_indices[0] + len(entity.token_indices)
        expected_last_index = entity.token_indices[-1] + 1
        assert actual_last_index == expected_last_index, \
            f'Brat only supports continuous entity spans, but found indices {entity.token_indices}'
        entity_start = sample.tokens[min(entity.token_indices)].start_char_index + text_offset
        entity_stop = sample.tokens[max(entity.token_indices)].stop_char_index + text_offset

        return f'T{entity_id}\tmention {entity_start} {entity_stop}\t{entity.text}\n'

    def _dumps_relation(self, relation: model.Relation, relation_id: int, head_id: int, tail_id: int) -> str:
        return f'R{relation_id}\t{relation.type} Arg1:T{head_id} Arg2:T{tail_id}\n'

    def _export_batch(self, base_name: str, samples: typing.List[model.Sample]) -> None:
        with open(f'{base_name}.txt', 'w', encoding='utf8') as txt_file, \
                open(f'{base_name}.ann', 'w', encoding='utf8') as ann_file:
            entity_id = 1
            relation_id = 1
            batch_text = ''
            text_offset = 0
            for sample in samples:
                text = f'{sample.text}\n'
                txt_file.write(text)
                batch_text += text
                for relation in sample.relations:
                    head_id = entity_id
                    tail_id = entity_id + 1

                    ann_file.write(self._dumps_entity(sample, relation.head, entity_id, batch_text, text_offset))
                    ann_file.write(self._dumps_entity(sample, relation.tail, entity_id, batch_text, text_offset))
                    ann_file.write(self._dumps_relation(relation, relation_id, head_id, tail_id))

                    entity_id = tail_id + 1
                    relation_id += 1
                text_offset += len(text)

    def _get_batch(self, samples: typing.List[model.Sample], batch_size: int):
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]

    def save(self, data_set: model.DataSet, file_path: str) -> None:
        if os.path.exists(file_path):
            # remove old directory, too make sure previously exported ann/txt
            # files are invalidated (in case they are not overwritten)
            shutil.rmtree(file_path)
        os.makedirs(file_path, exist_ok=True)

        for batch_id, batch in enumerate(self._get_batch(data_set.samples, self._num_sents_per_doc)):
            self._export_batch(os.path.join(file_path, f'{batch_id}'), batch)
