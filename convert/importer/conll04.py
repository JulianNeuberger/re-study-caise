import dataclasses
import typing

import stanza
import stanza.models.common.doc as stanza_model
import tqdm

import model
from convert import util
from convert.importer import base


class ConLL04Importer(base.BaseImporter):
    @dataclasses.dataclass
    class Block:
        tokens: typing.List[str]
        relations: typing.List[str]

    @dataclasses.dataclass
    class RawToken:
        text: str
        pos_tag: str
        ner_tag: str
        start_char: int
        stop_char: int

    def __init__(self, top_n: typing.Optional[int] = None, skip: typing.Optional[int] = 0):
        super().__init__(top_n, skip)
        stanza.download('en')
        self._nlp_pipeline = stanza.Pipeline(lang='en',
                                             processors='tokenize,mwt,pos,lemma,depparse',
                                             tokenize_pretokenized=True)

    @staticmethod
    def _get_sentences(file_path: str) -> typing.Iterator['ConLL04Importer.Block']:
        with open(file_path, 'r', encoding='utf8') as f:
            empty_lines_seen = 0
            tokens: typing.List[str] = []
            relations: typing.List[str] = []
            for line in tqdm.tqdm(f):
                line = line.strip()

                if line != '':
                    # either a token or a relation
                    is_relation = empty_lines_seen == 1
                    if is_relation:
                        relations.append(line)
                    else:
                        tokens.append(line)
                    continue

                empty_lines_seen += 1
                if empty_lines_seen == 2:
                    if len(tokens) == 0:
                        # no tokens mean no block, e.g. at the end of the file
                        continue
                    block = ConLL04Importer.Block(
                        tokens=tokens,
                        relations=relations
                    )
                    tokens = []
                    relations = []
                    empty_lines_seen = 0
                    yield block

    def _block_to_sample(self, block: 'ConLL04Importer.Block') -> model.Sample:
        assert len(block.tokens) > 0
        last_sample_id: typing.Optional[str] = None
        current_start_index = 0

        current_token_index = 0
        entities: typing.Dict[int, model.Entity] = {}
        raw_tokens: typing.List[ConLL04Importer.RawToken] = []
        for token_line in block.tokens:
            # 563	Org	17	O	NN/NNP/NNP/NNP/IN/NNPS	2nd/U.S./Circuit/Court/of/Appeals	O	O	O
            sample_id, ner_tag, entity_id, _, pos_tag, token_text, _, _, _ = token_line.split('\t')
            assert last_sample_id is None or last_sample_id == sample_id
            last_sample_id = sample_id

            is_entity = ner_tag != 'O'
            # multi token entities are denoted by their text/pos tags joined with slashes ("/"),
            # but there are also tokens that are just a slash or contain a slash
            # we can check this by looking only at the pos tag attribute, which is never just a slash
            is_multi_token = '/' in pos_tag

            if is_multi_token:
                token_text = token_text.split('/')
                pos_tag = pos_tag.split('/')
            else:
                token_text = [token_text]
                pos_tag = [pos_tag]

            token_indices: typing.List[int] = []
            token_texts: typing.List[str] = []
            token_ner_tags = [ner_tag for _ in range(len(token_text))]
            token_ner_tags = util.plain_ner_tags_to_bioes_tags(token_ner_tags)
            for t, p, token_ner in zip(token_text, pos_tag, token_ner_tags):
                assert '/' not in p

                if p == ',' and t == 'COMMA':
                    t = ','

                token_indices.append(current_token_index)
                stop_index = current_start_index + len(t)
                raw_tokens.append(ConLL04Importer.RawToken(
                    text=t,
                    ner_tag=token_ner,
                    pos_tag=p,
                    start_char=current_start_index,
                    stop_char=stop_index
                ))
                current_start_index = stop_index
                current_token_index += 1
                token_texts.append(t)

            if is_entity:
                entities[int(entity_id)] = model.Entity(
                    text=' '.join(token_texts),
                    token_indices=token_indices,
                    ner_tag=ner_tag
                )

        assert last_sample_id is not None

        doc: stanza_model.Document = self._nlp_pipeline([[t.text for t in raw_tokens]])
        tokens = self._build_tokens(doc)
        updated_tokens: typing.List[model.Token] = []
        for raw_token, token in zip(raw_tokens, tokens):
            updated_tokens.append(model.Token(
                text=token.text,
                ner_tag=raw_token.ner_tag,
                pos_tag=raw_token.pos_tag,
                start_char_index=token.start_char_index,
                stop_char_index=token.stop_char_index,
                dependency_relation=token.dependency_relation
            ))

        relations: typing.List[model.Relation] = []
        for raw_relation in block.relations:
            head_index, tail_index, type_text = raw_relation.split('\t')
            relations.append(model.Relation(
                type=type_text,
                head=entities[int(head_index)],
                tail=entities[int(tail_index)]
            ))

        return model.Sample(
            id=last_sample_id,
            text=' '.join([t.text for t in updated_tokens]),
            entities=list(entities.values()),
            relations=relations,
            tokens=updated_tokens
        )

    def load(self, file_paths: typing.List[str], name: typing.Optional[str] = None) -> model.DataSet:
        if name is None:
            name = file_paths[0]
        assert len(file_paths) == 1

        samples: typing.List[model.Sample] = []
        for block in ConLL04Importer._get_sentences(file_paths[0]):
            samples.append(self._block_to_sample(block))
        return model.DataSet(samples=samples, name=name)
