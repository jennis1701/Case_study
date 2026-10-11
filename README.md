# Explainable Bloom's Taxonomy Classification of Student Responses Using Large Language Models

> **M.Tech AI & Data Science Thesis Project**

---

## 1. Research Objective

This project investigates whether large language models (LLMs) can accurately classify the cognitive level demonstrated in a student's free-form response according to **Revised Bloom's Taxonomy**, and whether they can generate human-readable explanations that justify each classification.

The system goes beyond keyword matching. It uses structured prompt engineering with Gemma 3 to consider the *meaning* and *reasoning process* present in the student's response.

---

## 2. Features

- **Six-level Bloom classification** — Remember, Understand, Apply, Analyze, Evaluate, Create
- **Structured LLM output** — JSON with bloom level, confidence, evidence, and explanation
- **Explainability** — Evidence is grounded in the actual student response text
- **Light-themed academic UI** — Built with Streamlit
- **Configurable model** — Swap Gemma 3 variants or other models via `.env`
- **Experiment logging** — CSV log for research reproducibility
- **Evaluation module** — Accuracy, Precision, Recall, F1, Confusion Matrix (via scikit-learn)
- **Modular architecture** — Prompts, inference, parsing, and UI are all separated

---

## 3. Architecture

```
bloom-llm-thesis/
│
├── app.py                    ← Streamlit UI (entry point)
├── config.py                 ← Centralised configuration
├── requirements.txt
├── .env.example              ← Copy to .env before running
│
├── src/
│   ├── __init__.py
│   ├── model_service.py      ← Gemma 3 loading and inference
│   ├── bloom_classifier.py   ← Orchestrates prompt → inference → parse
│   ├── prompt_templates.py   ← All prompt templates
│   ├── response_parser.py    ← LLM output parsing and validation
│   ├── evaluation.py         ← Batch evaluation metrics
│   └── utils.py              ← Logging, colour maps, text sanitisation
│
├── data/                     ← Evaluation datasets (not committed)
├── experiments/              ← Inference logs and experiment results
└── tests/
    ├── test_parser.py        ← Parser unit tests
    └── test_bloom_classifier.py  ← Classifier unit tests (mock model)
```

**Data flow:**

```
User Input (question + response)
    │
    ▼
prompt_templates.py  →  Structured LLM Prompt
    │
    ▼
model_service.py     →  Gemma 3 (Hugging Face Transformers)
    │
    ▼
response_parser.py   →  ClassificationResult object
    │
    ▼
app.py               →  Streamlit UI result display
    │
    ▼
utils.py             →  Experiment log CSV
```

---

## 4. Revised Bloom's Taxonomy Levels

| Level | Description |
|---|---|
| **Remember** | Recall facts, definitions, terms, formulas, or previously learned information |
| **Understand** | Explain, interpret, summarise, describe, or demonstrate understanding of a concept |
| **Apply** | Use learned knowledge, a method, rule, formula, or procedure to solve a problem |
| **Analyze** | Break information into parts, identify relationships, compare components |
| **Evaluate** | Make a judgment, justify a decision, critique something, assess using criteria |
| **Create** | Generate, design, formulate, construct, propose, or produce something new |

---

## 5. Technology Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Gemma 3 (Hugging Face Transformers) |
| Inference | `transformers`, `torch`, `accelerate` |
| Data | `pandas`, `numpy` |
| Evaluation | `scikit-learn` |
| Configuration | `python-dotenv` |
| Language | Python 3.10+ |

---

## 6. Installation

### Prerequisites

- Python 3.10 or 3.11
- pip
- (Recommended) A CUDA-capable GPU with at least 6 GB VRAM for the 1B model

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/bloom-llm-thesis.git
cd bloom-llm-thesis
```

If you already have the project locally, simply open the folder and continue from the next step.

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```powershell
python -m pip install -r requirements.txt
```

**GPU users:** Install PyTorch with CUDA support first:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

**Important:** current Transformers releases require PyTorch 2.5+. If you see the error "PyTorch >= 2.5 is required", run:

```powershell
python -m pip install --upgrade "torch>=2.5.0"
```

### Step 4 — Configure environment

On Windows:

```powershell
Copy-Item .env.example .env
notepad .env
```

Edit `.env` and set:

```env
MODEL_NAME=google/gemma-3-1b-it
HF_TOKEN=hf_your_token_here
```

---

## 7. Gemma 3 Setup

Gemma 3 is a gated model on Hugging Face. You must:

1. Create a Hugging Face account: https://huggingface.co
2. Accept the Gemma 3 model licence:
   - https://huggingface.co/google/gemma-3-1b-it
3. Generate a Hugging Face access token:
   - https://huggingface.co/settings/tokens
4. Log in on your machine:

```bash
huggingface-cli login
```

Or add to `.env`:

```
HF_TOKEN=hf_your_token_here
```

### Model size recommendations

| Model | Parameters | VRAM Required | Notes |
|---|---|---|---|
| `google/gemma-3-1b-it` | 1B | ~3 GB | CPU-friendly, fastest |
| `google/gemma-3-4b-it` | 4B | ~8 GB | Good balance |
| `google/gemma-3-12b-it` | 12B | ~24 GB | High accuracy |
| `google/gemma-3-27b-it` | 27B | ~55 GB | Best quality |

---

## 8. Running the Application

Use the module form so it works reliably even when the `streamlit` launcher is not on `PATH`:

```powershell
python -m streamlit run app.py
```

The application opens at `http://localhost:8501`.

If `streamlit` still is not recognized, use:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

---

## 9. Running the Tests

```bash
python -m pytest tests/ -v
```

Tests use a mock model service and do not require a GPU or a downloaded model.

---

## 10. Example Input and Output

**Question:**
```
Compare the time complexity of bubble sort and merge sort and explain
why merge sort performs better on large datasets.
```

**Student Response:**
```
Bubble sort has O(n²) complexity because it compares all pairs in
each pass. Merge sort has O(n log n) because it divides the array
recursively, reducing comparisons. For large n, n log n grows much
more slowly, so merge sort is more efficient.
```

**Output:**
```json
{
  "bloom_level": "Analyze",
  "confidence": 0.88,
  "evidence": "The student identifies the trade-offs between time complexities
               and explains the cause of the difference.",
  "explanation": "The response demonstrates Analyze: the student breaks the
                  problem into components (two algorithms), examines the
                  cause of different complexities, and explains how they
                  relate to performance on large inputs.",
  "alternative_level": "Evaluate",
  "alternative_reason": "While the student draws a conclusion about efficiency,
                         there is no explicit value judgment or comparison
                         against criteria — the analysis stops at explanation
                         of the mechanism."
}
```

---

## 11. Evaluation Methodology

The labelled CSV is the source of truth. Each row must contain a question, a student response, and its human-assigned Bloom level. The model receives only the question and response; the label is held back and used only to score the prediction.

Required columns are `question`, `student_response`, and `bloom_level`. Column names are matched case-insensitively; `true_bloom_level` and `ground_truth_bloom_level` are also accepted as label-column names. Bloom labels are matched case-insensitively and normalized to the six canonical levels.

In the Streamlit app, upload the CSV under **Ground-Truth Dataset**. Uploading only loads and validates the reference data; it does not run the model over the full dataset. Select a CSV row and click **Load selected row into the form**, then click **Analyse Response**. The app uses the selected row's question and response as model inputs and its `bloom_level` as ground truth. The label is never sent to the model. Manually entered questions and responses are also scored when they match a normalized CSV pair. Conflicting CSV labels for a duplicate pair are reported as an error.

Accuracy, macro Precision/Recall/F1, and the confusion matrix accumulate over unique matched rows analyzed in the current app session. Invalid model outputs count as incorrect. Changing or removing the CSV clears those session scores. This workflow lets you inspect selected examples without classifying all 600 rows.

For a full-dataset batch evaluation, use the separate command-line runner:

```powershell
python evaluate.py data/my_dataset.csv `
    --predictions experiments/predictions.csv `
    --report experiments/evaluation_report.json
```

Reported metrics include accuracy over all dataset rows, macro Precision/Recall/F1 over the six Bloom levels, per-class metrics, and a confusion matrix. Invalid model outputs count as incorrect and appear in a separate `Invalid` prediction column. Valid-prediction coverage is reported separately. The prediction CSV includes the ground-truth label, predicted label, prediction status, and an `is_correct` value for each row.

---

## 12. Research Disclaimer

> This system provides an AI-assisted Bloom's Taxonomy classification for research purposes. The result should be reviewed by an educator and should not be treated as a definitive assessment of a student's ability.

Model-reported confidence is not a statistically calibrated probability. It is the LLM's self-assessed certainty and should be interpreted accordingly.

---

## 13. Future Work

- Calibration study of model confidence scores
- Comparison with TF-IDF + SVM and BERT-based classifiers
- Fine-tuned Bloom classifier training on labelled datasets
- Inter-rater agreement study with human expert annotations
- Multi-label classification (when a response spans multiple levels)
- Retrieval-Augmented Generation (RAG) with pedagogy literature

---

## 14. Citation

If you use this project in your research, please cite it as:

```
[Author Name]. (2025). Explainable Bloom's Taxonomy Classification of
Student Responses Using Large Language Models. M.Tech Thesis,
[University Name].
```
