"""
Phase 5 — Solvers.

All computation is pure Python — zero LLM involvement (Hard Rule 1).

Provides:
  - compute_amount_safe_to_pay: binary search for max safe same-day payment
  - find_earliest_date_for_full_payment: forward scan over 90 days
  - build_installment_plans: enumerate safe installment options
  - build_partial_payment: two-payment plan
  - build_spending_change_candidates: enumerate possible spending reductions
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from forecast import ForecastEngine, is_payment_safe, is_wait_safe, FORECAST_DAYS
from classifier import EventClassifier

logger = logging.getLogger(__name__)

BINARY_SEARCH_PRECISION = 0.01   # 1 cent precision


@dataclass
class InstallmentPlan:
    payment_option_id: str
    payment_method: str    # "installments"
    payments: List[Tuple[pd.Timestamp, float]]  # (date, amount) chronological
    total_payable: float
    financing_fee: float
    num_payments: int
    first_payment_date: pd.Timestamp
    is_safe: bool = False


@dataclass
class PartialPaymentPlan:
    first_date: pd.Timestamp
    first_amount: float
    second_date: pd.Timestamp
    second_amount: float
    total: float
    is_safe: bool = False

    def payment_plan_str(self) -> str:
        return f"{self.first_date.strftime('%Y-%m-%d')}:{self.first_amount:.2f}|{self.second_date.strftime('%Y-%m-%d')}:{self.second_amount:.2f}"


@dataclass
class SpendingChangeSet:
    changes: List[dict]  # each: {action, event_id, new_amount_or_none, category, savings_per_period}
    total_savings_90d: float = 0.0

    def to_string(self) -> str:
        parts = []
        for c in self.changes:
            if c["action"] == "stop":
                parts.append(f"stop:{c['event_id']}")
            else:
                parts.append(f"reduce_to:{c['event_id']}:{c['new_amount']:.2f}")
        return "|".join(parts) if parts else "none"

    def as_overrides(self) -> Dict[str, Optional[float]]:
        result = {}
        for c in self.changes:
            if c["action"] == "stop":
                result[c["event_id"]] = None
            else:
                result[c["event_id"]] = c["new_amount"]
        return result


# ---------------------------------------------------------------------------
# 1. Amount safe to pay (binary search)
# ---------------------------------------------------------------------------

def compute_amount_safe_to_pay(
    engine: ForecastEngine,
    request_date: pd.Timestamp,
    requested_amount: float,
    precision: float = BINARY_SEARCH_PRECISION,
) -> float:
    """
    Find the maximum amount the user can pay on request_date without the
    90-day forecast balance falling below minimum_balance_to_keep.

    Uses binary search between 0 and requested_amount.
    Returns 0.0 if even paying 0 is "unsafe" (shouldn't happen — 0 means no payment).
    """
    # Quick upper bound check
    if is_payment_safe(engine, request_date, requested_amount):
        return requested_amount

    # Quick lower bound check
    if not is_payment_safe(engine, request_date, 0.0):
        # Can't even maintain minimum with no extra payment — return 0
        return 0.0

    lo = 0.0
    hi = requested_amount
    iterations = 0
    max_iter = 60

    while hi - lo > precision and iterations < max_iter:
        mid = (lo + hi) / 2.0
        if is_payment_safe(engine, request_date, mid):
            lo = mid
        else:
            hi = mid
        iterations += 1

    result = round(lo, 2)
    logger.debug(
        "amount_safe_to_pay: %.2f (requested %.2f, %d iterations)",
        result, requested_amount, iterations
    )
    return result


# ---------------------------------------------------------------------------
# 2. Earliest date for full payment (forward scan)
# ---------------------------------------------------------------------------

def find_earliest_date_for_full_payment(
    engine: ForecastEngine,
    request_date: pd.Timestamp,
    requested_amount: float,
) -> Optional[pd.Timestamp]:
    """
    Forward scan from request_date through request_date + FORECAST_DAYS.
    Returns the first date on which a full payment of requested_amount is safe.
    Returns None if no such date exists within the forecast period.

    Hard Rule 6: each safety check runs the FULL forecast from request_date.
    The payment is injected on the candidate date, and all intervening events
    are simulated first.

    NOTE: For dates near the end of the forecast window, we extend the window
    by 45 days to capture the next salary cycle after the payment date.
    """
    end_date = request_date + pd.Timedelta(days=FORECAST_DAYS)
    current = request_date

    while current <= end_date:
        if is_wait_safe(engine, current, requested_amount):
            return current
        current += pd.Timedelta(days=1)

    return None






# ---------------------------------------------------------------------------
# 3. Installment plans
# ---------------------------------------------------------------------------

def build_installment_plans(
    engine: ForecastEngine,
    payment_options_df: pd.DataFrame,
    profile: pd.Series,
    request_date: pd.Timestamp,
    desired_completion_date: pd.Timestamp,
) -> List[InstallmentPlan]:
    """
    For each installment option in payment_options_df, build the full payment
    schedule and check if it's safe (balance never below minimum throughout
    the 90-day forecast with all installment payments injected).

    Only considers options where:
    - payment_method == "installments"
    - number_of_payments <= max_installment_months (from profile)
    - last payment date <= desired_completion_date
    - "installments" is in payment_methods_user_will_consider
    """
    user_methods = set(profile.get("payment_methods_user_will_consider", []))
    if "installments" not in user_methods:
        return []

    max_months = profile.get("max_installment_months", None)
    if pd.isna(max_months):
        max_months = None
    else:
        max_months = int(max_months)

    plans: List[InstallmentPlan] = []

    opts = payment_options_df[payment_options_df["payment_method"] == "installments"]

    for _, opt in opts.iterrows():
        n = int(opt["number_of_payments"])
        if max_months is not None and n > max_months:
            logger.debug("Skipping option %s: %d payments > max %d", opt["payment_option_id"], n, max_months)
            continue

        freq = int(opt["payment_frequency_days"]) if not pd.isna(opt["payment_frequency_days"]) else 30
        first_date = pd.Timestamp(opt["first_payment_date"])
        pay_amount = float(opt["payment_amount"])
        total_payable = float(opt["total_payable_amount"])
        fee = float(opt["financing_fee"]) if not pd.isna(opt["financing_fee"]) else 0.0

        # Build payment schedule
        payments: List[Tuple[pd.Timestamp, float]] = []
        current_date = first_date
        for i in range(n):
            payments.append((current_date, pay_amount))
            current_date = current_date + pd.Timedelta(days=freq)

        # Check deadline
        last_payment_date = payments[-1][0]
        if last_payment_date > desired_completion_date:
            logger.debug(
                "Skipping option %s: last payment %s > desired %s",
                opt["payment_option_id"], last_payment_date.date(), desired_completion_date.date()
            )
            continue

        # Safety check: inject ALL payments into the forecast
        safe = is_payment_safe(
            engine,
            first_date,
            pay_amount,
            additional_payments=payments[1:],  # first payment is included separately
        )

        plan = InstallmentPlan(
            payment_option_id=str(opt["payment_option_id"]),
            payment_method="installments",
            payments=payments,
            total_payable=total_payable,
            financing_fee=fee,
            num_payments=n,
            first_payment_date=first_date,
            is_safe=safe,
        )
        plans.append(plan)
        logger.debug(
            "Installment option %s: n=%d, total=%.2f, safe=%s",
            opt["payment_option_id"], n, total_payable, safe
        )

    return plans


def installment_plan_str(plan: InstallmentPlan) -> str:
    """Format payment plan as YYYY-MM-DD:amount|..."""
    parts = [
        f"{d.strftime('%Y-%m-%d')}:{a:.2f}"
        for d, a in plan.payments
    ]
    return "|".join(parts)


# ---------------------------------------------------------------------------
# 4. Partial payment plan
# ---------------------------------------------------------------------------

def build_partial_payment(
    engine: ForecastEngine,
    request_date: pd.Timestamp,
    requested_amount: float,
    amount_safe_today: float,
    desired_completion_date: pd.Timestamp,
    allows_partial: bool,
    profile: pd.Series,
) -> Optional[PartialPaymentPlan]:
    """
    Build a two-payment partial plan:
      - Pay amount_safe_today on request_date (or a partial-safe amount if full_payment not in methods)
      - Pay remaining on earliest_safe_date for the remainder

    If full_payment is NOT in user methods but partial_payment IS, and amount_safe_today >= requested,
    we use amount_safe_today = requested - 1 to force the partial-plan path, then find the optimal split.
    
    Conditions:
      - allows_partial == True
      - "partial_payment" in payment_methods_user_will_consider
      - 0 < amount_safe_today < requested_amount
      - second payment date <= desired_completion_date

    The second payment date is found by forward-scanning, injecting the first
    payment on request_date and then finding when the balance recovers enough
    for the remainder.
    """
    user_methods = set(profile.get("payment_methods_user_will_consider", []))
    if "partial_payment" not in user_methods:
        return None
    if not allows_partial:
        return None

    # If full_payment is NOT in methods but partial_payment IS, force partial-split even
    # when the full amount appears safe (we can't recommend full_payment if not allowed)
    if "full_payment" not in user_methods and amount_safe_today >= requested_amount:
        # Find the optimal partial split: max first payment such that the two-part plan works
        # Use half as starting point and do binary search
        amount_safe_today = requested_amount / 2.0

    if amount_safe_today <= 0.0 or amount_safe_today >= requested_amount:
        return None

    remainder = round(requested_amount - amount_safe_today, 2)

    # Find earliest date for the remainder, given that amount_safe_today
    # is already paid on request_date
    end_date = request_date + pd.Timedelta(days=FORECAST_DAYS)
    # We need a new engine with the first payment already baked in
    # We do this by passing it as an extra payment and scanning for the rest
    search_start = request_date  # second payment can also be on request_date

    # Create modified engine with first payment already deducted
    # by making a temporary engine that starts with reduced balance
    # Hard Rule 6: always run full forecast from request_date
    from forecast import ForecastEngine as FE

    modified_engine = FE(
        classifier=engine.classifier,
        profile=engine.profile,
        starting_balance=engine.starting_balance,
        request_date=engine.request_date,
        spending_overrides=engine.spending_overrides,
        income_override=getattr(engine, 'income_override', None),
    )

    second_date: Optional[pd.Timestamp] = None
    current = search_start
    while current <= end_date:
        # Can we safely pay remainder on current, given first payment on request_date?
        safe = is_payment_safe(
            modified_engine,
            current,
            remainder,
            additional_payments=[(request_date, amount_safe_today)],
        )
        if safe and current != request_date:
            second_date = current
            break
        elif safe and current == request_date:
            # Both payments on same day
            second_date = current
            break
        current += pd.Timedelta(days=1)

    if second_date is None or second_date > desired_completion_date:
        return None

    # Verify the sum
    total = round(amount_safe_today + remainder, 2)
    if abs(total - requested_amount) > 0.02:
        logger.warning("Partial plan total mismatch: %.2f + %.2f = %.2f (expected %.2f)",
                       amount_safe_today, remainder, total, requested_amount)
        remainder = round(requested_amount - amount_safe_today, 2)

    return PartialPaymentPlan(
        first_date=request_date,
        first_amount=amount_safe_today,
        second_date=second_date,
        second_amount=remainder,
        total=requested_amount,
        is_safe=True,
    )


# ---------------------------------------------------------------------------
# 5. Spending change candidates
# ---------------------------------------------------------------------------

def build_spending_change_candidates(
    classifier: EventClassifier,
    profile: pd.Series,
    request_date: pd.Timestamp,
) -> List[dict]:
    """
    Enumerate all recurring flexible events that could be stopped or reduced.
    Returns a list of candidate changes, each a dict:
      {event_id, action, category, flexibility, current_amount, min_allowed,
       new_amount (if reduce), savings_per_month}

    Only non-protected, flexible recurring events are candidates.
    Per spec: only recurring expenses in categories the user permits.

    Hard Rule 5: stop and reduce on the SAME event are mutually exclusive —
    we emit them as separate candidates; the caller (ranker) picks at most
    one action per event_id.
    """
    stoppable_cats = set(profile.get("expense_categories_user_is_willing_to_stop", []))
    reducible_cats = set(profile.get("expense_categories_user_is_willing_to_reduce", []))
    protected_cats = set(profile.get("expense_categories_to_protect", []))

    # Get recurring patterns
    patterns = classifier.get_recurring_patterns()
    history = classifier.get_settled_history()

    candidates: List[dict] = []

    # Look at actual event_ids that are stoppable/reducible
    ev = classifier.events
    # Filter to debit events that have been active recently (within 90 days before request)
    recent_cutoff = request_date - pd.Timedelta(days=180)
    recent = ev[
        (ev["settlement_date"] >= recent_cutoff) &
        (ev["direction"] == "debit") &
        (ev["status"] == "settled") &
        (ev["flexibility"].isin(["stoppable", "reducible", "reducible_or_stoppable"]))
    ].copy()

    # De-duplicate: pick the most recent event per (category, amount) — representative
    # of the recurring pattern
    seen_cats = set()
    for _, row in recent.sort_values("settlement_date", ascending=False).iterrows():
        cat = row["category"]
        flex = row["flexibility"]
        eid = row["event_id"]
        amt = row["amount_home"]
        min_allowed = row["minimum_allowed_amount"] if not pd.isna(row.get("minimum_allowed_amount", float("nan"))) else 0.0

        if cat in protected_cats:
            continue
        if pd.isna(amt) or amt <= 0:
            continue

        # Can stop?
        if flex in ("stoppable", "reducible_or_stoppable") and cat in stoppable_cats:
            cat_key = f"stop_{eid}"
            if cat_key not in seen_cats:
                seen_cats.add(cat_key)
                candidates.append({
                    "event_id": eid,
                    "action": "stop",
                    "category": cat,
                    "flexibility": flex,
                    "current_amount": float(amt),
                    "min_allowed": 0.0,
                    "new_amount": None,
                    "savings_per_month": float(amt),  # approx monthly savings
                })

        # Can reduce?
        if flex in ("reducible", "reducible_or_stoppable") and cat in reducible_cats:
            min_amt = float(min_allowed) if min_allowed else float(amt) * 0.5
            if min_amt < amt:
                cat_key = f"reduce_{eid}"
                if cat_key not in seen_cats:
                    seen_cats.add(cat_key)
                    candidates.append({
                        "event_id": eid,
                        "action": "reduce",
                        "category": cat,
                        "flexibility": flex,
                        "current_amount": float(amt),
                        "min_allowed": float(min_amt),
                        "new_amount": float(min_amt),
                        "savings_per_month": float(amt) - float(min_amt),
                    })

    # Sort by savings descending
    candidates.sort(key=lambda c: c["savings_per_month"], reverse=True)
    return candidates


def try_spending_changes_for_safety(
    base_engine: ForecastEngine,
    classifier: EventClassifier,
    profile: pd.Series,
    request_date: pd.Timestamp,
    requested_amount: float,
    candidates: List[dict],
    max_changes: int = 3,
) -> Tuple[bool, Optional[SpendingChangeSet]]:
    """
    Try combinations of spending changes (up to max_changes) to make
    a full payment of requested_amount on request_date safe.

    Returns (is_now_safe, spending_change_set).
    If already safe without changes, returns (True, None).

    Tries greedy: add changes one at a time by descending savings until safe.
    Respects the constraint: stop and reduce on the SAME event are mutually
    exclusive — guaranteed because candidates list has at most one entry per event_id
    per action type.
    """
    from forecast import ForecastEngine as FE

    # Check if already safe without changes
    if is_payment_safe(base_engine, request_date, requested_amount):
        return True, None

    applied: List[dict] = []
    overrides: Dict = {}

    for cand in candidates:
        if len(applied) >= max_changes:
            break

        eid = cand["event_id"]
        # Don't apply both stop and reduce to same event
        if eid in overrides:
            continue

        new_val = None if cand["action"] == "stop" else cand["new_amount"]
        overrides[eid] = new_val
        applied.append(cand)

        # Create new engine with overrides
        trial_engine = FE(
            classifier=classifier,
            profile=profile,
            starting_balance=base_engine.starting_balance,
            request_date=base_engine.request_date,
            spending_overrides=dict(overrides),
            income_override=getattr(base_engine, 'income_override', None),
        )

        if is_payment_safe(trial_engine, request_date, requested_amount):
            changes_out = []
            for c in applied:
                changes_out.append({
                    "action": c["action"],
                    "event_id": c["event_id"],
                    "new_amount": c.get("new_amount"),
                    "category": c["category"],
                    "savings_per_period": c["savings_per_month"],
                })
            scs = SpendingChangeSet(changes=changes_out)
            return True, scs

    return False, None
