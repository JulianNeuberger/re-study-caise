Download file at https://cogcomp.seas.upenn.edu/Data/ER/conll04.corp and store 
in this folder.

Run `python -m main convert -f data/input/conll04/conll04.corp -o data/export/conll04/jsonl/ --in-format conll --out-format line-json --split "train.json:.7" --split "valid.json:.1" --split "test.json:.2" --shuffle --seed 42`
