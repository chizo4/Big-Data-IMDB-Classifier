# Big-Data-IMDB-Classifier

TODO (sections):
- note: completed as part of uva course
- about (in a few sentences)
- classic contribution

### WIP: instructions

1. to run code - clone repo, access, etc.

```bash
git clone git@github.com:chizo4/Big-Data-IMDB-Classifier.git
```

2. make sure to have ollama installed via (todo:). install models via:

```bash
ollama pull [model_name]
```

in this project we utilize the `gemma3:1b` model:

```bash
ollama pull gemma3:1b
```

3. first setup conda env. this creates a stable python 3.10 environment and install all respective dependencies from `requirements.txt` (all experiments are run via this conda environment).
```bash
bash script/setup-env.sh
```

4. to run training script execute:
```bash
bash script/run-train.sh imdb directing.json writing.json gemma3:1b
```

5. run prediction on DEV set:
```bash
bash script/run-predict imdb directing.json writing.json validation_hidden.csv gemma3:1b
```

6. run prediction on TEST set:
```bash
bash script/run-predict imdb directing.json writing.json test_hidden.csv gemma3:1b
```
