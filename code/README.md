# Buy or Wait? — Solution

## Setup

```bash
pip install pandas numpy python-dotenv groq openai tenacity Pillow pytesseract sentence-transformers
```

Copy `.env.example` to `.env` and add your `GROQ_API_KEY` (get free at console.groq.com).

## Run

```bash
# Full run (250 requests → output.csv)
python code/main.py

# Full run with LLM (message/image extraction + explanations)
python code/main.py  # set GROQ_API_KEY in .env

# Calibration against sample_requests.csv (25 rows, no LLM)
python code/main.py --sample --no-llm

# Single request debug
python code/main.py --request request_01 --sample --no-llm
```

## Architecture

All financial computation is pure Python (zero LLM in the decision path):

1. **loaders.py** — CSV loaders + currency conversion (settlement-date rate lookup)
2. **retrieval.py** — context builder (hard ID-filter joins)
3. **classifier.py** — recurring event detection (by category/flexibility, never description strings)
4. **forecast.py** — 90-day day-by-day ledger
5. **solvers.py** — binary search (amount_safe_to_pay), forward scan (earliest_date), installment/partial builders
6. **ranker.py** — 6-step tiebreak ranker with strict eligibility from `payment_methods_user_will_consider`
7. **validator.py** — output schema enforcement
8. **extraction.py** — Tesseract OCR → text LLM → vision LLM fallback (with tenacity retry, status-aware cache)
9. **explainer.py** — explanation writer with self-consistency check
10. **pipeline.py** — single-request orchestrator
11. **main.py** — CLI entry point

## Output

`output.csv` with columns:
```
request_id,amount_safe_to_pay,affordability_status,recommended_payment_method,
payment_plan,earliest_date_for_full_payment,spending_changes_needed,decision_explanation
```
