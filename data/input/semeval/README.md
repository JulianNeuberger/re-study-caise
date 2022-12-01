Download from http://www.kozareva.com/downloads.html (SemEval 2010 Task 8) and extract archive,
so that this folder contains 
- SemEval2010_task8_scorer-v1.2
- SemEval2010_task8_testing
- SemEval2010_task8_testing_keys
- SemEval2010_task8_training

Run `python -m main convert -f data/input/semeval/SemEval2010_task8_training/TRAIN_FILE.TXT -o data/export/semeval/jsonl --in-format semeval --out-format line-json --split "train.json:.8" --split "valid.json:.2" --shuffle --seed 42`

Run `python -m main convert -f data/input/semeval/SemEval2010_task8_testing_keys/TEST_FILE_FULL.txt -o data/export/semeval/jsonl/test.json --in-format semeval --out-format line-json`