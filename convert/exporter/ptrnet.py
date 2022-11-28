import json
import os

import typing

import tqdm

import model
from convert.exporter import base


class PtrNetExporter(base.BaseExporter):
    @staticmethod
    def _write_sent_file(base_dir: str, subset_name: str, dataset: model.DataSet):
        with open(os.path.join(base_dir, f'{subset_name}.sent'), 'w', encoding='utf8') as f:
            for s in dataset.samples:
                f.write(f'{" ".join([t.text for t in s.tokens])}\n')

    @staticmethod
    def _write_tuple_file(base_dir: str, subset_name: str, dataset: model.DataSet):
        with open(os.path.join(base_dir, f'{subset_name}.tup'), 'w', encoding='utf8') as f:
            for s in dataset.samples:
                serialized_relations = [
                    f'{r.head.text} ; {r.tail.text} ; {r.type}'
                    for r in s.relations
                ]
                f.write(f'{" | ".join(serialized_relations)}\n')

    @staticmethod
    def _write_pointer_file(base_dir: str, subset_name: str, dataset: model.DataSet):
        with open(os.path.join(base_dir, f'{subset_name}.pointer'), 'w', encoding='utf8') as f:
            for s in dataset.samples:
                serialized_pointers = [
                    f'{min(r.head.token_indices)} {max(r.head.token_indices)} {min(r.tail.token_indices)} {max(r.head.token_indices)} {r.type}'
                    for r in s.relations
                ]
                f.write(f'{" | ".join(serialized_pointers)}\n')

    @staticmethod
    def _get_dependency_distances(start_index: int, tokens: typing.List[model.Token]):
        distances = {
            start_index: 0
        }

        while len(distances) < len(tokens):
            # up the tree
            for token_index, token in enumerate(tokens):
                root, _ = token.dependency_relation
                if root is None:
                    continue
                if token_index in distances:
                    continue
                if root in distances:
                    distances[token_index] = distances[root] + 1

            # down the tree
            items = list(distances.items())
            for known_index, known_distance in items:
                root, _ = tokens[known_index].dependency_relation
                if root is None:
                    continue
                if root in distances:
                    continue
                distances[root] = known_distance + 1

        return [d for i, d in sorted(distances.items(), key=lambda entry: entry[0])]

    @staticmethod
    def _write_dependency_file(base_dir: str, subset_name: str, dataset: model.DataSet):
        with open(os.path.join(base_dir, f'{subset_name}.dep'), 'w', encoding='utf8') as f:
            for s in tqdm.tqdm(dataset.samples, desc='Calculating and writing dependency adjacency matrices'):
                deps = [PtrNetExporter._get_dependency_distances(i, s.tokens) for i, _ in enumerate(s.tokens)]
                f.write(json.dumps({'adj_mat': deps}))
                f.write('\n')

    def save(self, dataset: model.DataSet, file_path: str) -> None:
        base_dir = os.path.dirname(file_path)
        set_name = os.path.basename(file_path)

        os.makedirs(base_dir, exist_ok=True)

        self._write_sent_file(base_dir, set_name, dataset)
        self._write_pointer_file(base_dir, set_name, dataset)
        self._write_tuple_file(base_dir, set_name, dataset)
        self._write_dependency_file(base_dir, set_name, dataset)
