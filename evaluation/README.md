# PDF RAG Assistant V2 Evaluation Framework

This directory contains the 40-question benchmark dataset and automated evaluation script for measuring retrieval precision, section discrimination, citation accuracy, and abstention accuracy.

---

## 📊 Dataset Overview (`questions.json`)

The evaluation dataset contains 40 carefully structured questions spanning 7 categories:

| Category | Questions | Objective |
| :--- | :--- | :--- |
| `direct_factual_retrieval` | 10 | Direct factual retrieval of definitions and formulas |
| `semantic_paraphrased` | 5 | Paraphrased questions testing semantic search |
| `multi_section_reasoning` | 5 | Questions requiring context across multiple sub-layers |
| `numerical_threshold` | 5 | Exact numerical value and hyperparameter extraction |
| `exception_negation` | 5 | Questions testing negative constraints and rules |
| `similar_section_discrimination` | 5 | Discrimination between related sections (includes Q08 mandatory regression test) |
| `out_of_document` | 5 | Verification of refusal behavior (`"I couldn't find the answer..."`) |

---

## 🏃 Running Evaluation

To run the full benchmark comparing **Baseline (Embedding-only)** vs. **V2 (Embedding + CrossEncoder Reranker)**:

```bash
python -m evaluation.run_evaluation --mode both
```

To run only V2 mode:

```bash
python -m evaluation.run_evaluation --mode v2
```

To run only Baseline mode:

```bash
python -m evaluation.run_evaluation --mode baseline
```
