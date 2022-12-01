Download from https://drive.google.com/file/d/1jt_yurHorliIor8uDqvirQlGyFYwW81c/view and unzip files
in this folder.

Run `python -m main convert -f data/input/nyt10/train.json -o data/export/nyt10/jsonl --in-format nyt10 --out-format line-json --split "train.json:.8" --split "valid.json:.2" --shuffle --seed 42`

Run `python -m main convert -f data/input/nyt10/test.json -o data/export/nyt10/jsonl/test.json --in-format nyt10 --out-format line-json`