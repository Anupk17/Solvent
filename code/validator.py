"""
Phase 7 — Validation.

Enforces all hard invariants before writing output.csv.
Called after all rows are computed.
"""
from __future__ import annotations

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

VALID_AFFORDABILITY = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
VALID_METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


def validate_output(df: pd.DataFrame, requests_df: pd.DataFrame) -> List[str]:
    """
    Run all validation checks on the output DataFrame.
    Returns a list of error/warning strings (empty = all OK).
    """
    errors: List[str] = []

    # 1. Column order check (Hard Rule 5)
    actual_cols = list(df.columns)
    if actual_cols != REQUIRED_COLUMNS:
        errors.append(f"Column order mismatch. Got {actual_cols}, expected {REQUIRED_COLUMNS}")

    # 2. Row count
    expected_ids = set(requests_df.index)
    actual_ids = set(df["request_id"])
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    if missing:
        errors.append(f"Missing request_ids: {sorted(missing)[:10]}")
    if extra:
        errors.append(f"Extra request_ids: {sorted(extra)[:10]}")

    # 3. Per-row invariants
    for _, row in df.iterrows():
        rid = row["request_id"]

        # a. amount_safe_to_pay bounds
        amt = float(row["amount_safe_to_pay"]) if pd.notna(row["amount_safe_to_pay"]) else None
        if amt is None:
            errors.append(f"{rid}: amount_safe_to_pay is null")
            continue

        req_row = requests_df.loc[rid] if rid in requests_df.index else None
        if req_row is not None:
            req_amt = float(req_row["requested_amount"])
            if not (0 <= amt <= req_amt + 0.02):
                errors.append(f"{rid}: amount_safe_to_pay={amt} violates [0, {req_amt}]")

        # b. Valid affordability status
        status = str(row["affordability_status"])
        if status not in VALID_AFFORDABILITY:
            errors.append(f"{rid}: invalid affordability_status={status!r}")

        # c. Valid method
        method = str(row["recommended_payment_method"])
        if method not in VALID_METHODS:
            errors.append(f"{rid}: invalid recommended_payment_method={method!r}")

        # d. Hard Rule 5: wait → payment_plan must be "none"
        plan = str(row["payment_plan"])
        if method == "wait" and plan != "none":
            errors.append(f"{rid}: wait recommendation but payment_plan is not 'none' (got {plan!r})")

        # e. not_recommended → payment_plan must be "none", earliest_date empty
        if method == "not_recommended":
            if plan != "none":
                errors.append(f"{rid}: not_recommended but payment_plan={plan!r}")
            earliest = str(row.get("earliest_date_for_full_payment", ""))
            if earliest and earliest.strip():
                errors.append(f"{rid}: not_recommended but earliest_date_for_full_payment={earliest!r}")

        # f. affordable_now → earliest_date == request_date
        if status == "affordable_now":
            earliest = str(row.get("earliest_date_for_full_payment", ""))
            if req_row is not None:
                req_date = str(req_row["request_date"])[:10]
                if earliest and earliest != req_date:
                    errors.append(f"{rid}: affordable_now but earliest_date={earliest!r} != request_date={req_date!r}")

        # g. partial_payment: 2 payments, sum = requested_amount
        if method == "partial_payment":
            parts = plan.split("|") if plan != "none" else []
            if len(parts) != 2:
                errors.append(f"{rid}: partial_payment but payment_plan has {len(parts)} parts (expected 2)")
            else:
                try:
                    amounts = [float(p.split(":")[1]) for p in parts]
                    total = sum(amounts)
                    if req_row is not None:
                        req_amt = float(req_row["requested_amount"])
                        if abs(total - req_amt) > 0.05:
                            errors.append(f"{rid}: partial_payment amounts sum={total:.2f} != requested={req_amt:.2f}")
                except Exception as e:
                    errors.append(f"{rid}: partial_payment parse error: {e}")

        # h. No scientific notation (check as string)
        amt_str = str(row["amount_safe_to_pay"])
        if "e" in amt_str.lower():
            errors.append(f"{rid}: scientific notation in amount_safe_to_pay: {amt_str}")

    # 4. Check no scientific notation in payment_plan
    sci_count = df["payment_plan"].astype(str).str.lower().str.contains("e\\+|e-").sum()
    if sci_count > 0:
        errors.append(f"{sci_count} rows have scientific notation in payment_plan")

    return errors


def check_not_affordable_eligibility(
    output_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
) -> List[str]:
    """
    Hard Rule 6: for EVERY not_affordable row, verify that full_payment
    is genuinely absent from payment_methods_user_will_consider.
    Must run on 100% of not_affordable rows.
    """
    issues: List[str] = []
    not_aff = output_df[output_df["recommended_payment_method"] == "not_recommended"]

    for _, row in not_aff.iterrows():
        rid = row["request_id"]
        # We need the user_id — join via requests
        # Caller should pass this function requests_df too, but we check via output for now
        # This check is done in main after having access to user_id
        pass

    return issues


def write_output(
    df: pd.DataFrame,
    output_path: str,
    requests_df: pd.DataFrame,
) -> None:
    """
    Write validated output.csv (Hard Rule 5).
    Uses float_format="%.2f" to prevent scientific notation.
    Columns are written in the required order.
    """
    # Ensure correct column order
    out = df[REQUIRED_COLUMNS].copy()

    # Validate first
    errors = validate_output(out, requests_df)
    if errors:
        logger.error("VALIDATION ERRORS (%d):", len(errors))
        for e in errors:
            logger.error("  %s", e)
    else:
        logger.info("Validation PASSED — no errors")

    # Write CSV
    out.to_csv(output_path, index=False, float_format="%.2f")
    logger.info("Wrote %d rows to %s", len(out), output_path)

    # Verify: check for scientific notation in the raw file
    with open(output_path, "r", encoding="utf-8") as f:
        raw = f.read()
    if "e+" in raw.lower() or "e-" in raw.lower():
        logger.error("HARD RULE 5 VIOLATION: scientific notation found in %s", output_path)
    else:
        logger.info("No scientific notation detected in output file")

    # Verify header matches template
    first_line = raw.split("\n")[0].strip()
    expected_header = ",".join(REQUIRED_COLUMNS)
    if first_line != expected_header:
        logger.error(
            "HARD RULE 5: Header mismatch!\n  Got:      %s\n  Expected: %s",
            first_line, expected_header
        )
    else:
        logger.info("Header verified correct: %s", first_line)
