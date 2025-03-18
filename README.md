# Big-Data-IMDB-Classifier

TODO (sections):
- note: completed as part of uva course
- about (in a few sentences)
- classic contribution

data source: https://github.com/hazourahh/big-data-course-2024-projects

### WIP: instructions

1. to run code - clone repo, access, etc.

```bash
git clone git@github.com:chizo4/Big-Data-IMDB-Classifier.git
```

2. make sure to have ollama installed via (todo:). install models via:

```bash
ollama pull [model_name]
```

in this project we utilize the `gemma3:4b` model:

```bash
ollama pull gemma3:4b
```

3. first setup conda env. this creates a stable python 3.10 environment and install all respective dependencies from `requirements.txt` (all experiments are run via this conda environment).
```bash
bash script/setup-env.sh
```

4. to run training script execute (takes a while):
```bash
bash script/run-train.sh imdb directing.json writing.json gemma3:4b
```

note: since llm operations to create synthetic data is very have we apply a caching approach, storing the genre prediction in csv after first run on train. thanks to this, another run on save join data with these predictions to avoid this heavy recomputation.

5. run prediction on DEV set:
```bash
bash script/run-predict.sh imdb directing.json writing.json validation_hidden.csv gemma3:4b
```

note: similarly as per training, we cache the first run predictions when performing feature engineering here

6. run prediction on TEST set:
```bash
bash script/run-predict.sh imdb directing.json writing.json test_hidden.csv gemma3:4b
```

note: similarly as per training, we cache the first run predictions when performing feature engineering here
