# Enterprise Agentic RAG Platform

## Evaluation

The baseline evaluation dataset is grounded in the three PDFs in `data/`: `encryption_google_cloud.pdf`, `google_cloud_security.pdf`, and `microsoft_annual_report_2025.pdf`. Its questions and source references are maintained in `evaluation/eval_dataset.py`; the auditable ground-truth PDF is generated from that same dataset with:

```bash
python -m evaluation.generate_ground_truth_pdf
```

Run the RAGAS evaluation against the configured live RAG pipeline with:

```bash
python -m evaluation.evaluate
```

RAGAS reports faithfulness (whether answers are supported by retrieved context), answer relevancy (whether answers address the question), context precision (whether retrieved context is useful), and context recall (whether retrieved context covers the ground-truth answer). Aggregate scores are written to `evaluation/latest_scores.json`; answers, contexts, sources, available agent metadata, and per-question metric values are written to `evaluation/latest_results.json`.

The evaluation dataset and generated artifacts remain under `evaluation/`. Ingestion reads only PDF files directly under `data/`, so evaluation questions, ground truths, reports, and the ground-truth PDF are not included in the searchable corpus.
