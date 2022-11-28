import json
import os
import shutil
import typing
import tqdm


def replace_rel_id_with_name(dir_name: str):
    def replace_ids_in_file(file_path: str):
        shutil.copy(file_path, f'{file_path}.bkp')
        with open(file_path, 'r', encoding='utf8') as original_file:
            original_data = json.load(original_file)
        # [["H", 6, 7], ["T", 8, 9], 3]
        for kind in tqdm.tqdm(['predictions', 'target'], desc=file_path):
            for sample in original_data[kind]:
                for relation in sample:
                    _, _, relation_id = relation
                    relation[2] = id2rel[relation_id]
        with open(file_path, 'w', encoding='utf8') as target_file:
            json.dump(original_data, target_file)

    rel2id_file_path = os.path.join(dir_name, 'rel2id.json')
    with open(rel2id_file_path, 'r', encoding='utf8') as f:
        rel2id = json.load(f)
    id2rel = {v: k for k, v in rel2id.items()}

    replace_ids_in_file(os.path.join(dir_name, 'dev-results.json'))
    replace_ids_in_file(os.path.join(dir_name, 'test-results.json'))


if __name__ == '__main__':
    for d in ['conll04', 'fewrel', 'nyt10', 'semeval']:
        replace_rel_id_with_name(d)
