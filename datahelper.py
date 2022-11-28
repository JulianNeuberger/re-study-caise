import os

datasets = [
    'conll04',
    'fewrel',
    'nyt10',
    'semeval',
]
out_formats = {
    'casrel': {'train': 'train_triples.json', 'test': 'test_triples.json', 'valid': 'dev_triples.json'},
    'joint-er': {'train': 'train.json', 'test': 'test.json', 'valid': 'dev.json'},
    'rsan': {'train': 'train.json', 'test': 'test.json', 'valid': 'dev.json'},
    'spert': {'train': 'train.json', 'test': 'test.json', 'valid': 'dev.json'},
    'two-are-better-than-one': {'train': 'train.json', 'test': 'test.json', 'valid': 'dev.json'},
    'pfn': {'train': 'train.json', 'test': 'test.json', 'valid': 'valid.json'},
    'mare': {'train': 'train.jsonl', 'test': 'test.jsonl', 'valid': 'dev.jsonl'},
}

for dataset in datasets:
    for out_format, subsets in out_formats.items():
        for source, target in subsets.items():
            os.system(f'python -m main convert -f data/export/{dataset}/jsonl/{source}.json -o data/export/{dataset}/{out_format}/{target} --in-format line-json --out-format {out_format}')
