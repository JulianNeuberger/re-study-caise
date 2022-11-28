import json
import typing

import tqdm

import model
from convert.importer import base


class TacredImporter(base.BaseImporter):
    def load(self, file_paths: typing.List[str], name: typing.Optional[str] = None) -> model.DataSet:
        assert len(file_paths) == 1
        with open(file_paths[0], 'r', encoding='utf8') as f:
            data = json.load(f)

        samples: typing.List[model.Sample] = []
        for raw_sample in tqdm.tqdm(data):
            raw_tokens: typing.List[str] = raw_sample['token']
            pos_tags: typing.List[str] = raw_sample['stanford_pos']
            ner_tags: typing.List[str] = raw_sample['stanford_ner']
            dep_rels: typing.List[str] = raw_sample['stanford_deprel']
            dep_heads: typing.List[int] = raw_sample['stanford_head']

            tokens = []
            start_char = 0
            for text, pos, ner, dep_rel, dep_head in zip(raw_tokens, pos_tags, ner_tags, dep_rels, dep_heads):
                stop_char = start_char + len(text)
                dep_head -= 1
                if dep_head == -1:
                    dep_head = None
                token = model.Token(
                    text=text,
                    ner_tag=ner,
                    pos_tag=pos,
                    dependency_relation=(dep_head, dep_rel),
                    start_char_index=start_char,
                    stop_char_index=stop_char
                )
                start_char = stop_char
                # account for whitespace
                start_char += 1

                tokens.append(token)

            head_indices = list(range(raw_sample['subj_start'], raw_sample['subj_end'] + 1))
            head = model.Entity(
                text=' '.join([raw_tokens[i] for i in head_indices]),
                token_indices=head_indices
            )

            tail_indices = list(range(raw_sample['obj_start'], raw_sample['obj_end'] + 1))
            tail = model.Entity(
                text=' '.join([raw_tokens[i] for i in tail_indices]),
                token_indices=tail_indices
            )

            relations: typing.List[model.Relation] = [
                model.Relation(
                    type=raw_sample['relation'],
                    head=head,
                    tail=tail
                )
            ]
            entities: typing.List[model.Entity] = [head, tail]

            sample_id = raw_sample['id']
            text = ' '.join(raw_tokens)

            sample = model.Sample(
                id=sample_id,
                text=text,
                tokens=tokens,
                entities=entities,
                relations=relations
            )
            samples.append(sample)

        return model.DataSet(samples=samples, name=file_paths[0])
