import json
import typing

import click
import stanza
import stanza.models.common.doc as stanza_model
import tqdm

import model
from convert import util
from convert.exporter.base import LineJsonExporter
from convert.importer import base


class FewRelImporter(base.BaseImporter):
    def __init__(self, top_n: int = None, skip: int = 0):
        # TODO: implement skipping
        super().__init__(top_n, skip)
        stanza.download('en')
        self._nlp_pipeline = stanza.Pipeline(lang='en',
                                             processors='tokenize,mwt,pos,lemma,depparse,ner',
                                             tokenize_pretokenized=True)

    def load(self, file_paths: typing.List[str], name: typing.Optional[str] = None) -> model.DataSet:
        if name is None:
            name = file_paths[0]
        assert len(file_paths) == 2

        data_file_path, relation_names_file_path = file_paths

        with open(data_file_path, 'r', encoding='utf8') as f2:
            file = json.load(f2)
            sample_id = 0
            parse_error_counter = 0
            samples = []
            for key in tqdm.tqdm(file.keys(), desc='Relation types'):
                for sample in tqdm.tqdm(file[key], leave=False, desc=f'Samples for "{key}"'):
                    token_list = []

                    # indices for head and tail are zero-indexed!
                    entity_list = []
                    head_entity_text = sample['h'][0]
                    head_entity_indices = sample['h'][2][0]

                    tail_entity_text = sample['t'][0]
                    tail_entity_indices = sample['t'][2][0]

                    text = ''
                    token_number = 0
                    for token in sample['tokens']:
                        # check for problematic tokens in the samples
                        if len(token) == 0 or token.isspace():
                            # reduce entity ids if they appear after the problematic token
                            if token_number < head_entity_indices[0]:
                                head_entity_indices = [x - 1 for x in head_entity_indices]
                            # two separate if-statements because of python short-circuiting
                            if token_number < tail_entity_indices[0]:
                                tail_entity_indices = [x - 1 for x in tail_entity_indices]
                            continue  # do not add the offending token to the text string
                        token_number = token_number + 1
                        text = text + token + ' '

                    try:
                        raw_tokens = text.strip().split(' ')
                        doc: stanza_model.Document = self._nlp_pipeline([raw_tokens])
                        stanza_output = base.BaseImporter._build_tokens(doc)

                        for token in stanza_output:
                            token_list.append(model.Token(text=token.text,
                                                          start_char_index=token.start_char_index,
                                                          stop_char_index=token.stop_char_index,
                                                          ner_tag=token.ner_tag,
                                                          pos_tag=token.pos_tag,
                                                          dependency_relation=token.dependency_relation))
                    except:
                        parse_error_counter = parse_error_counter + 1
                        pass

                    head_entity_tokens = [token_list[i] for i in head_entity_indices]
                    tail_entity_tokens = [token_list[i] for i in tail_entity_indices]

                    head_entity = model.Entity(text=head_entity_text,
                                               token_indices=head_entity_indices,
                                               ner_tag=util.plain_ner_tag_from_bioes(head_entity_tokens[0].ner_tag))
                    tail_entity = model.Entity(text=tail_entity_text,
                                               token_indices=tail_entity_indices,
                                               ner_tag=util.plain_ner_tag_from_bioes(tail_entity_tokens[0].ner_tag))
                    entity_list.append(head_entity)
                    entity_list.append(tail_entity)

                    relation_list = []
                    # relation_head = head_entity
                    # relation_type = tail_entity

                    # the file pid2name.json contains a dictionary with the relation abbreviations and their respective
                    # description. As the samples are sorted according to the relations, the description is extracted
                    # and inserted into our data format accordingly
                    with open(relation_names_file_path, 'r', encoding='utf8') as f1:
                        relation_file = json.load(f1)
                        relation_type = relation_file.get(str(key))[0]

                    relation = model.Relation(type=relation_type, head=head_entity, tail=tail_entity)
                    relation_list.append(relation)

                    self._fix_ner_tags(token_list, entity_list, relation_list)

                    samples.append(model.Sample(id=sample_id, text=text, entities=entity_list, tokens=token_list,
                                                relations=relation_list))
                    sample_id = sample_id + 1
            click.echo(f'{parse_error_counter} out of {sample_id} could not be parsed correctly. '
                       f'Error quote is {parse_error_counter / sample_id}')

            # parse_error_file.close()
            return model.DataSet(samples=samples, name=name)


if __name__ == '__main__':
    res = FewRelImporter().load(['../../data/input/FewRel/train.json', '../../data/input/FewRel/pid2name.json'])
    LineJsonExporter().save(res, 'data/export/few_rel/train.json')
