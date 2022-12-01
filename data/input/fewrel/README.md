Download the files 

1) pid2name.json
2) train_wiki.json
3) val_wiki.json

from https://github.com/thunlp/FewRel/tree/master/data and store them in this directory.

Run `python -m main convert -f data/input/FewRel/train_wiki.json -f data/input/FewRel/pid2name.json -o data/export/FewRel/jsonl --in-format fewrel --out-format line-json --split "train.json:.8" --split "valid.json:.2" --shuffle --seed 42`

Run `python -m main convert -f data/input/FewRel/val_wiki.json -f data/input/FewRel/pid2name.json -o data/export/FewRel/jsonl/test.json --in-format fewrel --out-format line-json`

Inside `data/export/FewRel/jsonl` run `python concat.py`

In root directory run `python -m main convert -f data/export/fewrel/jsonl/full.json -o data/export/fewrel/jsonl/ --in-format line-json --out-format line-json --shuffle --seed 42 --split "train.json:.7" --split "test.json:.2" --split "valid.json:.1" --stratified`
