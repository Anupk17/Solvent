# SOLVENT — Buy or Wait? Task Tracker

Single source of truth. Re-verify against reality at the start of every session.

## Status Legend
- `[ ]` not started
- `[~]` in progress
- `[x]` complete — must have pasted real evidence

---

## Phase 0 — Setup
- [~] Write SESSION START log.txt entry
- [ ] Create venv, install all deps
- [ ] Verify imports: pandas, numpy, PIL, pytesseract, tenacity, groq/openai, sentence_transformers

## Phase 1 — Loaders
- [ ] Load financial_profiles.csv
- [ ] Load financial_events.csv
- [ ] Load exchange_rates.csv
- [ ] Load requests.csv / sample_requests.csv
- [ ] Load request_payment_options.csv
- [ ] Load messages.csv
- [ ] Load images.csv
- [ ] Currency conversion with settlement-date rate lookup (verified on test case)

## Phase 2 — Retrieval
- [ ] Hard ID-filter joins (user→events, request→options, request→messages, request→images)
- [ ] Semantic re-rank (sentence-transformers, local, no API key)

## Phase 3 — Event Classification
- [ ] Recurring vs one-time via linked_event_id chains + category (NEVER description strings)
- [ ] Flexible vs essential from profile data
- [ ] Log projected recurring events per user (amount + dates) — inspectable

## Phase 4 — Forecast Engine
- [ ] Day-by-day ledger, 90 days from request_date
- [ ] Ignore: pending credits, failed/cancelled, duplicate/linked, unrealized
- [ ] Reserve pending debits
- [ ] Count settled salary on settlement_date only
- [ ] Sanity check: flag if forecast income = $0 or <<historical income
- [ ] Log forecast events per user

## Phase 5 — Solvers
- [ ] amount_safe_to_pay via binary search
- [ ] earliest_date_for_full_payment via forward scan
- [ ] Partial payment builder (exactly 2 payments summing to requested_amount)
- [ ] Installment plan matcher (max_installment_months enforced)
- [ ] Spending-change builder (only flexible, non-protected, ≤3 changes, stop+reduce mutually exclusive on same event)

## Phase 6 — Ranker
- [ ] Eligibility from payment_methods_user_will_consider ONLY
- [ ] wait eligible only if full_payment in list AND safe future date exists
- [ ] not_recommended fallback
- [ ] 6-step tiebreak (deadline → no spending changes → min total → earliest → fewer payments → lowest option_id)
- [ ] VERIFY: every not_affordable row has full_payment absent from payment_methods_user_will_consider

## Phase 7 — Validation
- [ ] Schema: exact column order, no extras
- [ ] wait → payment_plan = "none" (not empty, not the date)
- [ ] 0 ≤ amount_safe_to_pay ≤ requested_amount
- [ ] Partial payment: 2 payments sum = requested_amount
- [ ] No scientific notation in CSV
- [ ] Diff header against dataset/output.csv

## Phase 8 — Extraction (LLM)
- [ ] Tesseract OCR on every image first
- [ ] Text LLM to extract structured facts from OCR text
- [ ] Vision LLM fallback only if OCR yields nothing usable
- [ ] Cache with status: "ok" / "failed" — NEVER cache failed as ok
- [ ] tenacity retry/backoff on 429s
- [ ] Message extraction (structured amendments)
- [ ] Never treat blank amount as zero

## Phase 9 — Explanation Writer
- [ ] LLM writes explanation from already-computed decision (no computation inside)
- [ ] Self-consistency check: parse numbers from explanation, verify against decision
- [ ] Fallback to template if check fails twice

## Phase 10 — Calibration Gate
- [ ] Run against sample_requests.csv (25 rows)
- [ ] Report EXACT match count (e.g. "19/25") — never "mostly working"
- [ ] Per-mismatch trace: full financial state, forecast, candidates, specific logic line
- [ ] Iterate until 25/25 or fully defended near-25/25
- [ ] Note if any fix causes count to DROP (regression alert)

## Phase 11 — Full Run + Submission Package
- [ ] Run python code/main.py against full dataset/requests.csv (250 rows)
- [ ] Consistency scan: schema, wait/none, no sci-notation, not_affordable eligibility check on 100% of rows
- [ ] evaluation/usage_report.md
- [ ] .env.example with placeholders
- [ ] code.zip packaging
- [ ] Confirm output.csv in repo root with 250+1 rows and correct header

---

## Calibration History
| Run | Match Count | Notes |
|-----|------------|-------|
| (none yet) | — | — |

---

## Bug Log
| Date | Hard Rule | Bug | Fix |
|------|-----------|-----|-----|
| (none yet) | — | — | — |
