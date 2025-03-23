## **Big-Data-IMDB-Classifier** 🍿

### **About**
This project implements a high-performance movie classification pipeline using `Apache Spark`. It features a modular architecture that efficiently processes large IMDB datasets, applying efficient feature engineering including LLM-based genre prediction. The system uses a Random Forest classifier with optimized hyperparameters to predict movie ratings with accuracy exceeding 75%. Our implementation emphasizes consistent state management between training and prediction phases, efficient caching of expensive operations, and scalable processing of movie metadata from diverse sources. Data for this project is sourced from the [UvA Big Data Course 2024 Projects](https://github.com/hazourahh/big-data-course-2024-projects).

---

### **Implementation Notes**

The pipeline implements several optimizations to improve performance:
- [X] **LLM Caching:** Our system caches LLM-generated genre predictions to avoid expensive recomputation. After the first run, subsequent executions use the cached predictions.
- [X] **Unified Pipeline:** Training and prediction occur in a single pipeline to ensure consistent feature transformations and categorical mappings.
- [X] **Efficient Feature Engineering:** The pipeline handles complex categorical features through careful string indexing and applies appropriate scaling to numeric features.

---

### **Running Instructions**

### 1. Prerequisites
- Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Install [ollama](https://ollama.com/download)

---

### 2. Clone the Repository
```bash
git clone git@github.com:chizo4/Big-Data-IMDB-Classifier.git
cd Big-Data-IMDB-Classifier
```

---

### 3. Install the LLM Model

```bash
ollama pull gemma3:4b
```

---

### 4. Setup Environment

```bash
bash script/setup-env.sh
```

This creates a stable Python `3.10` environment and installs all dependencies from `requirements.txt`.

---

### 5. Run the Pipeline

(A) For `validation` set:

```bash
bash script/run-pipeline.sh imdb directing.json writing.json validation_hidden.csv gemma3:4b TMDB_movie_dataset_v11.csv
```

(B) For `test` set:

```bash
bash script/run-pipeline.sh imdb directing.json writing.json test_hidden.csv gemma3:4b TMDB_movie_dataset_v11.csv
```

---

### **Contribution**

> [!NOTE]
> In case you had any questions associated with this project, or spotted any issue (including technical setup), please feel free to contact [`Filip J. Cierkosz`](https://github.com/chizo4) via any of the links included in the profile page. You might also want to open an [`Issue`](https://github.com/chizo4/JusTreeAI/issues/new?template=Blank+issue) with any suggested fixes, or questions.
