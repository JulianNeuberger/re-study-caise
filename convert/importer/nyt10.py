import dataclasses
import json
import typing

import stanza
import stanza.models.common.doc as stanza_model

import model
from convert import util
from convert.exporter import LineJsonExporter
from convert.importer import base


@dataclasses.dataclass
class RawDocument:
    stanza_doc: stanza_model.Document
    raw_data: typing.Dict[str, typing.Any]


class Nyt10Importer(base.BaseImporter):
    def __init__(self, top_n: typing.Optional[int] = None, skip: int = 0, batch_size: int = 256):
        super().__init__(top_n, skip)
        stanza.download('en')
        self._nlp_pipeline = stanza.Pipeline(lang='en',
                                             processors='tokenize,mwt,pos,lemma,depparse,ner',
                                             tokenize_pretokenized=True)
        self._batch_size = batch_size

    @staticmethod
    def _build_entity_from_relation_data(r, entity_id: str, tokens: typing.List[model.Token]) -> model.Entity:
        assert entity_id in ('em1', 'em2')
        # from https://github.com/truthless11/HRL-RE:
        # 1, 4 == Inside, Begin of source entity
        # 2, 5 == Inside, Begin of target entity
        tags = (1, 4) if entity_id == 'em1' else (2, 5)

        e_text = r[entity_id]
        e_token_indices = []
        found_start = False
        for index, tag in enumerate(r['tags']):
            if tag not in tags:
                continue

            if tag == tags[1]:
                # beginning of entity
                e_token_indices.append(index)
                found_start = True
                continue

            # inside of entity
            if not found_start:
                # we found the inside of an entity before we found the start
                # this is true for some samples in https://github.com/truthless11/HRL-RE
                # fix it, by ignoring this tag
                continue

            # inside entity and we found the start
            e_token_indices.append(index)

        e_tokens = [tokens[i] for i in e_token_indices]
        return model.Entity(text=e_text,
                            token_indices=e_token_indices,
                            ner_tag=util.plain_ner_tag_from_bioes(e_tokens[0].ner_tag))

    def _collect_raw_documents(self, file: typing.IO, num_docs: int) -> typing.List[RawDocument]:
        docs: typing.List[RawDocument] = []
        while len(docs) < num_docs:
            line = file.readline()
            if line == '':
                # end of file
                break
            data = json.loads(line)
            text = data['sentext']
            raw_tokens = text.strip().split(' ')
            # use pretokenized sentence, e.g. pass data as list of sentences, with nested lists of tokens
            doc: stanza_model.Document = stanza.Document([], text=[raw_tokens])
            docs.append(RawDocument(doc, data))
        return docs

    def load(self, file_paths: typing.List[str], name: typing.Optional[str] = None) -> model.DataSet:
        if name is None:
            name = file_paths[0]
        assert len(file_paths) == 1

        samples = []

        with open(file_paths[0], 'r', encoding='utf8') as f:
            progress_bar = self._progress_bar(f)
            num_samples = 0

            while True:
                if self._skip and num_samples <= self._skip:
                    num_samples += 1
                    self._advance_progress_bar(progress_bar, f)
                    continue

                next_batch_size = self._batch_size
                if self._top_n:
                    num_samples_without_skipped = num_samples - self._skip
                    next_batch_size = min(next_batch_size, self._top_n - num_samples_without_skipped)
                    assert next_batch_size >= 0
                    assert next_batch_size <= self._batch_size

                raw_docs = self._collect_raw_documents(f, next_batch_size)
                raw_data_dicts = [r.raw_data for r in raw_docs]
                if len(raw_docs) == 0:
                    # end of file
                    break
                docs = self._nlp_pipeline([r.stanza_doc for r in raw_docs])

                for doc, data in zip(docs, raw_data_dicts):
                    if 'ID' in data:
                        sample_id = str(data['ID'])
                    else:
                        sample_id = str(num_samples)
                    text = data['sentext']

                    relations: typing.List[model.Relation] = []
                    entities: typing.Set[model.Entity] = set()
                    tokens = base.BaseImporter._build_tokens(doc)

                    for r in data['relations']:
                        e1 = self._build_entity_from_relation_data(r, 'em1', tokens)
                        e2 = self._build_entity_from_relation_data(r, 'em2', tokens)
                        entities.add(e1)
                        entities.add(e2)
                        r_type = r['rtext']
                        relations.append(model.Relation(head=e1, tail=e2, type=r_type))

                    entities_as_list = list(entities)
                    self._fix_ner_tags(tokens, entities_as_list, relations)

                    sample = model.Sample(
                        id=sample_id,
                        text=text,
                        entities=entities_as_list,
                        relations=relations,
                        tokens=tokens
                    )
                    samples.append(sample)

                    num_samples += 1

                self._advance_progress_bar(progress_bar, f, n_samples=next_batch_size)
            progress_bar.close()

        return model.DataSet(samples=samples, name=name)


if __name__ == '__main__':
    res = Nyt10Importer(top_n=2500, skip=33_000).load(['data/input/nyt10/train.json'])
    print(max([len(s.tokens) for s in res.samples]))
    LineJsonExporter().save(res, 'data/export/nyt10/train.json')
