import re
import typing

import stanza
import stanza.models.common.doc as stanza_model
import tqdm

import model
from convert import util
from convert.exporter import LineJsonExporter
from convert.importer import base


class SemEvalImporter(base.BaseImporter):
    """
    Class to contain the SemEval Task 8 dataset
    Data download: http://www.kozareva.com/downloads.html
    Paper download: https://arxiv.org/pdf/1911.10422.pdf

    One dataset consists of three lines with different info, see:
    1 "The system [...] application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>."
    Component-Whole(e2,e1)
    Comment: Not a collection: there is structure here, organisation.
    """

    def __init__(self, top_n: typing.Optional[int] = None, skip: int = 0):
        super().__init__(top_n, skip)
        stanza.download('en')
        self._nlp_pipeline = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse,ner')

    @staticmethod
    def _find_entity_char_indices(text: str) -> typing.Tuple[int, int, int, int]:
        e1_start = text.index('<e1>')
        e1_stop = text.index('</e1>')

        e2_start = text.index('<e2>')
        e2_stop = text.index('</e2>')

        # compensate for removal of start tags (4 chars) at their respective stop tags
        e1_stop -= 4
        e2_stop -= 4

        if e1_start < e2_start:
            # e1 is the first entity in the string
            # compensate for removing the <e1> tag (4 chars)
            e2_start -= 4
            e2_stop -= 4

            # compensate for removing the </e1> stop tag (5 chars)
            e2_start -= 5
            e2_stop -= 5
        else:
            # e2 is the first entity in the string
            e1_start -= 4
            e1_stop -= 4

            e1_start -= 5
            e1_stop -= 5
        return e1_start, e1_stop, e2_start, e2_stop

    def _process_sample(self, lines: typing.List[str]) -> model.Sample:
        text: str
        relation: str
        text, relation, _ = lines
        sample_id, text = text.split('\t', maxsplit=1)

        # strip quotes
        text = text[1:-1]

        # there are samples where the entity tags have no whitespace left/right of them
        # fix those samples, otherwise we have entities, that have no corresponding tokens
        text = re.sub(r'(\w)<e1>', r'\1 <e1>', text)
        text = re.sub(r'</e1>([\w.])', r'</e1> \1', text)
        text = re.sub(r'(\w)<e2>', r'\1 <e2>', text)
        text = re.sub(r'</e2>([\w.])', r'</e2> \1', text)

        e1_start, e1_stop, e2_start, e2_stop = self._find_entity_char_indices(text)

        text = text.replace('<e1>', '').replace('</e1>', '').replace('<e2>', '').replace('</e2>', '')

        doc: stanza_model.Document = self._nlp_pipeline(text)
        tokens = base.BaseImporter._build_tokens(doc)

        e1_tokens = [i for i, t in enumerate(tokens) if t.start_char_index >= e1_start and t.stop_char_index <= e1_stop]
        e2_tokens = [i for i, t in enumerate(tokens) if t.start_char_index >= e2_start and t.stop_char_index <= e2_stop]

        e1 = model.Entity(text=text[e1_start:e1_stop],
                          token_indices=e1_tokens,
                          ner_tag=util.plain_ner_tag_from_bioes(tokens[e1_tokens[0]].ner_tag))
        e2 = model.Entity(text=text[e2_start:e2_stop],
                          token_indices=e2_tokens,
                          ner_tag=util.plain_ner_tag_from_bioes(tokens[e2_tokens[0]].ner_tag))

        if relation == 'Other':
            relations = []
        else:
            relation_type, order = relation.split('(', maxsplit=1)
            if order.startswith('e1'):
                head = e1
                tail = e2
            else:
                head = e2
                tail = e1

            relations = [model.Relation(type=relation_type, head=head, tail=tail)]

        entities = [e1, e2]
        self._fix_ner_tags(tokens, entities, relations)

        return model.Sample(
            id=sample_id,
            text=text,
            tokens=tokens,
            entities=entities,
            relations=relations
        )

    def load(self, file_paths: typing.List[str], name: typing.Optional[str] = None) -> model.DataSet:
        if name is None:
            name = file_paths[0]
        assert len(file_paths) == 1, 'SemEval dataset splits only contain one file'
        samples = []

        with open(file_paths[0], 'r', encoding='utf8') as f:
            chunk = []
            for line in tqdm.tqdm(f):
                line = line.strip()
                if line == '':
                    # this marks the end of a chunk
                    # (or some random new lines at the end of the file)
                    if len(chunk) == 3:
                        # one chunk should be ready for processing
                        samples.append(self._process_sample(chunk))
                    chunk = []
                    continue

                # a line containing data
                chunk.append(line)

                if self._top_n is not None and len(samples) == self._top_n:
                    break
        return model.DataSet(samples=samples, name=name)


if __name__ == '__main__':
    res = SemEvalImporter(top_n=2500).load(['data/input/semeval/SemEval2010_task8_training/TRAIN_FILE.TXT'])
    print(max([len(s.tokens) for s in res.samples]))
    LineJsonExporter().save(res, 'data/export/semeval/train.json')
