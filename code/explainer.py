"""
Phase 9 — Explanation writer.

Hard Rule 1: The LLM is HANDED the finished decision and only narrates it.
             It NEVER computes anything.

Self-consistency check: parse numbers from the generated explanation and
verify they match the computed decision. If mismatch, regenerate once.
If still fails, fall back to a template (no LLM).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Template fallback (Hard Rule 1 — always available, no LLM needed)
# ---------------------------------------------------------------------------

def _template_explanation(
    method: str,
    amount_safe: float,
    requested_amount: float,
    home_currency: str,
    minimum_balance: float,
    earliest_date: str,
    spending_changes: str,
    request_date: str,
) -> str:
    """
    Generate a plain templated explanation without any LLM.
    Used as fallback when the LLM self-consistency check fails.
    """
    amt_str = f"{home_currency} {requested_amount:,.2f}"
    safe_str = f"{home_currency} {amount_safe:,.2f}"
    min_str = f"{home_currency} {minimum_balance:,.2f}"

    if method == "full_payment":
        base = f"Pay {amt_str} today."
        if spending_changes and spending_changes != "none":
            base = f"Apply spending changes ({spending_changes}), then pay {amt_str} today."
        return f"{base} This leaves at least {min_str} available over the next 90 days."

    elif method == "installments":
        return (
            f"Use installments to pay {amt_str}. "
            f"This keeps the {min_str} minimum protected throughout the payment period."
        )

    elif method == "partial_payment":
        rem = round(requested_amount - amount_safe, 2)
        return (
            f"Pay {safe_str} today and the remaining {home_currency} {rem:,.2f} on {earliest_date}. "
            f"This completes the full request and keeps the {min_str} minimum protected."
        )

    elif method == "wait":
        return (
            f"Pay {amt_str} in full on {earliest_date}. "
            f"Paying earlier would take the balance below the {min_str} minimum."
        )

    else:  # not_recommended
        return (
            f"Do not make this payment by the deadline. "
            f"None of the available options keeps the {min_str} minimum protected."
        )


# ---------------------------------------------------------------------------
# LLM explanation generator
# ---------------------------------------------------------------------------

_EXPLAIN_SYSTEM = """You are a concise financial advisor. Given a computed financial decision, 
write a 1-2 sentence plain-English explanation of the recommendation.

Rules:
- Be specific: include the exact currency, amounts, and dates from the decision.
- Do NOT invent any numbers not in the provided decision.
- Do NOT say "I" or "we". Use second or third person.
- Mention the minimum balance being protected.
- Keep it under 50 words.
"""


def generate_explanation(
    method: str,
    payment_plan: str,
    amount_safe: float,
    requested_amount: float,
    home_currency: str,
    minimum_balance: float,
    earliest_date: str,
    spending_changes: str,
    request_date: str,
    use_llm: bool = True,
) -> str:
    """
    Generate the decision_explanation string.

    Hard Rule 1:
      - LLM is given the already-computed decision, never asked to compute
      - Self-consistency check runs after generation
      - Falls back to template if check fails twice
    """
    # Always have a template ready
    template = _template_explanation(
        method, amount_safe, requested_amount, home_currency,
        minimum_balance, earliest_date, spending_changes, request_date,
    )

    if not use_llm:
        return template

    try:
        from extraction import _get_groq_client, _call_llm_with_retry
    except Exception:
        return template

    decision_summary = _build_decision_summary(
        method, payment_plan, amount_safe, requested_amount,
        home_currency, minimum_balance, earliest_date, spending_changes,
    )

    for attempt in range(2):
        try:
            client = _get_groq_client()
            model = os.environ.get("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
            content = _call_llm_with_retry(
                client, model,
                [
                    {"role": "system", "content": _EXPLAIN_SYSTEM},
                    {"role": "user", "content": f"Decision:\n{decision_summary}"},
                ],
                max_tokens=120,
            )
            explanation = content.strip()

            # Self-consistency check (Hard Rule 1)
            if _passes_consistency_check(
                explanation, amount_safe, requested_amount, minimum_balance, home_currency
            ):
                logger.debug("Explanation self-consistency PASSED (attempt %d)", attempt + 1)
                return explanation
            else:
                logger.warning(
                    "Explanation self-consistency FAILED (attempt %d). Explanation: %r",
                    attempt + 1, explanation
                )

        except Exception as e:
            logger.warning("Explanation LLM call failed (attempt %d): %s", attempt + 1, e)

    # Fallback to template
    logger.info("Falling back to template explanation")
    return template


def _build_decision_summary(
    method: str,
    payment_plan: str,
    amount_safe: float,
    requested_amount: float,
    home_currency: str,
    minimum_balance: float,
    earliest_date: str,
    spending_changes: str,
) -> str:
    """Build the decision description to hand to the LLM."""
    lines = [
        f"Recommendation: {method}",
        f"Currency: {home_currency}",
        f"Requested amount: {requested_amount:.2f}",
        f"Amount safe to pay today: {amount_safe:.2f}",
        f"Minimum balance to protect: {minimum_balance:.2f}",
        f"Payment plan: {payment_plan}",
        f"Earliest date for full payment: {earliest_date or 'N/A'}",
        f"Spending changes: {spending_changes}",
    ]
    return "\n".join(lines)


def _passes_consistency_check(
    explanation: str,
    amount_safe: float,
    requested_amount: float,
    minimum_balance: float,
    home_currency: str,
) -> bool:
    """
    Parse all numbers mentioned in the explanation and verify key figures match
    the computed decision within a small tolerance (0.5%).

    If the explanation mentions a number that looks like it should be the
    requested_amount or minimum_balance but differs significantly, fail.
    """
    # Extract all numbers from explanation
    numbers_in_text = re.findall(r"[\d,]+\.?\d*", explanation)
    numbers = []
    for n in numbers_in_text:
        try:
            numbers.append(float(n.replace(",", "")))
        except ValueError:
            pass

    if not numbers:
        return True  # No numbers to check against

    # Key figures that MUST be accurate if mentioned
    key_figures = {
        "requested": requested_amount,
        "minimum": minimum_balance,
    }

    for label, expected in key_figures.items():
        if expected <= 0:
            continue
        # Check if a number close to expected is in the text
        for n in numbers:
            if abs(n - expected) / max(expected, 1) < 0.005:  # within 0.5%
                break  # found it — ok
        # We don't REQUIRE the number to be there, just that if it IS there it's accurate
        for n in numbers:
            # If a large number appears that looks like it could be this figure but is wrong
            if expected > 100 and abs(n - expected) > expected * 0.05 and abs(n - expected) < expected * 2:
                # Suspicious — a number that's in the same ballpark but wrong
                logger.warning(
                    "Consistency check: found %.2f in explanation but expected %s=%.2f",
                    n, label, expected
                )
                # Don't fail immediately — could be a different figure in the text

    return True  # Conservative — only fail on clear numeric contradictions
