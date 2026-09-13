"""
Phase 6 — Ranker.

Selects the best recommendation from candidate plans using the 6-step tiebreak.
Eligibility is strictly enforced from payment_methods_user_will_consider (Hard Rule 6).

Tiebreak order (from spec):
  1. Complete the full request by desired_completion_date
  2. Require no spending changes
  3. Minimize total amount paid
  4. Start payment earlier
  5. Use fewer payments
  6. Use the lowest payment_option_id (as string sort) as final tiebreaker

Hard Rule 6:
  - full_payment / partial_payment / installments are eligible ONLY if
    they appear in payment_methods_user_will_consider
  - wait is eligible ONLY if full_payment is in that list AND a safe future date exists
  - not_recommended is the fallback when no eligible safe plan exists
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from forecast import ForecastEngine, is_payment_safe, FORECAST_DAYS
from solvers import (
    InstallmentPlan, PartialPaymentPlan, SpendingChangeSet,
    installment_plan_str,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output row dataclass
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    request_id: str
    amount_safe_to_pay: float
    affordability_status: str          # affordable_now | affordable_with_plan | affordable_later | not_affordable
    recommended_payment_method: str    # full_payment | partial_payment | installments | wait | not_recommended
    payment_plan: str                  # YYYY-MM-DD:amount|... or "none"
    earliest_date_for_full_payment: str  # YYYY-MM-DD or ""
    spending_changes_needed: str       # "none" or pipe-separated changes
    decision_explanation: str          # filled in Phase 9


# ---------------------------------------------------------------------------
# Candidate plan dataclass
# ---------------------------------------------------------------------------

@dataclass
class CandidatePlan:
    method: str                     # full_payment | partial_payment | installments | wait
    completes_by_deadline: bool
    spending_changes: Optional[SpendingChangeSet]
    total_paid: float
    first_payment_date: pd.Timestamp
    num_payments: int
    option_id: str                  # for tiebreak; "" for non-installment plans
    payment_plan_str: str
    earliest_full_date: Optional[pd.Timestamp]
    amount_safe_today: float
    # Extra data for building output
    installment_plan: Optional[InstallmentPlan] = None
    partial_plan: Optional[PartialPaymentPlan] = None


def _sort_key(plan: CandidatePlan) -> Tuple:
    """6-step tiebreak as a sort key (ascending = better)."""
    return (
        0 if plan.completes_by_deadline else 1,    # 1: completes by deadline
        0 if plan.spending_changes is None else 1,  # 2: no spending changes
        round(plan.total_paid, 2),                  # 3: minimize total paid
        plan.first_payment_date,                    # 4: start earlier
        plan.num_payments,                          # 5: fewer payments
        plan.option_id,                             # 6: lowest option_id
    )


# ---------------------------------------------------------------------------
# Core ranker
# ---------------------------------------------------------------------------

def rank_and_select(
    request_id: str,
    request: pd.Series,
    profile: pd.Series,
    engine: ForecastEngine,
    amount_safe_today: float,
    earliest_full_date: Optional[pd.Timestamp],
    installment_plans: List[InstallmentPlan],
    partial_plan: Optional[PartialPaymentPlan],
    spending_change_full: Optional[SpendingChangeSet],
    spending_change_full_engine: Optional[ForecastEngine],
) -> Recommendation:
    """
    Build candidate plans and pick the best one.

    Parameters:
      amount_safe_today         — computed by binary search (no spending changes)
      earliest_full_date        — computed by forward scan (no spending changes)
      installment_plans         — list of safe installment options (already safety-checked)
      partial_plan              — safe two-payment plan or None
      spending_change_full      — spending changes that unlock full payment today, or None
      spending_change_full_engine — engine with those overrides applied
    """
    user_methods: set = set(profile.get("payment_methods_user_will_consider", []))
    requested_amount: float = float(request["requested_amount"])
    request_date: pd.Timestamp = pd.Timestamp(request["request_date"])
    desired_date: pd.Timestamp = pd.Timestamp(request["desired_completion_date"])
    allows_partial: bool = bool(request["allows_partial_payment"])

    candidates: List[CandidatePlan] = []

    # --- 1. Full payment today (no spending changes) ---
    if "full_payment" in user_methods:
        if amount_safe_today >= requested_amount:
            candidates.append(CandidatePlan(
                method="full_payment",
                completes_by_deadline=(request_date <= desired_date),
                spending_changes=None,
                total_paid=requested_amount,
                first_payment_date=request_date,
                num_payments=1,
                option_id="",
                payment_plan_str=f"{request_date.strftime('%Y-%m-%d')}:{requested_amount:.2f}",
                earliest_full_date=request_date,
                amount_safe_today=amount_safe_today,
            ))

    # --- 2. Full payment today WITH spending changes ---
    if "full_payment" in user_methods and spending_change_full is not None and spending_change_full_engine is not None:
        if is_payment_safe(spending_change_full_engine, request_date, requested_amount):
            candidates.append(CandidatePlan(
                method="full_payment",
                completes_by_deadline=(request_date <= desired_date),
                spending_changes=spending_change_full,
                total_paid=requested_amount,
                first_payment_date=request_date,
                num_payments=1,
                option_id="",
                payment_plan_str=f"{request_date.strftime('%Y-%m-%d')}:{requested_amount:.2f}",
                earliest_full_date=request_date,
                amount_safe_today=amount_safe_today,
            ))

    # --- 3. Installment plans ---
    if "installments" in user_methods:
        for inst in installment_plans:
            if not inst.is_safe:
                continue
            completes = inst.payments[-1][0] <= desired_date
            candidates.append(CandidatePlan(
                method="installments",
                completes_by_deadline=completes,
                spending_changes=None,
                total_paid=inst.total_payable,
                first_payment_date=inst.first_payment_date,
                num_payments=inst.num_payments,
                option_id=inst.payment_option_id,
                payment_plan_str=installment_plan_str(inst),
                earliest_full_date=earliest_full_date,
                amount_safe_today=amount_safe_today,
                installment_plan=inst,
            ))

    # --- 4. Partial payment ---
    if "partial_payment" in user_methods and allows_partial and partial_plan is not None and partial_plan.is_safe:
        completes = partial_plan.second_date <= desired_date
        candidates.append(CandidatePlan(
            method="partial_payment",
            completes_by_deadline=completes,
            spending_changes=None,
            total_paid=requested_amount,  # partial pays full requested, no fee
            first_payment_date=request_date,
            num_payments=2,
            option_id="",
            payment_plan_str=partial_plan.payment_plan_str(),
            earliest_full_date=earliest_full_date,
            amount_safe_today=amount_safe_today,
            partial_plan=partial_plan,
        ))

    # --- 5. Wait (full_payment in future) ---
    if "full_payment" in user_methods and earliest_full_date is not None:
        if earliest_full_date > request_date:
            completes = earliest_full_date <= desired_date
            candidates.append(CandidatePlan(
                method="wait",
                completes_by_deadline=completes,
                spending_changes=None,
                total_paid=requested_amount,
                first_payment_date=earliest_full_date,
                num_payments=1,
                option_id="",
                payment_plan_str=f"{earliest_full_date.strftime('%Y-%m-%d')}:{requested_amount:.2f}",
                earliest_full_date=earliest_full_date,
                amount_safe_today=amount_safe_today,
            ))

    # --- Sort candidates ---
    candidates.sort(key=_sort_key)

    # --- Format earliest_date_for_full_payment ---
    earliest_full_str = ""
    if earliest_full_date is not None:
        earliest_full_str = earliest_full_date.strftime("%Y-%m-%d")

    # --- Build output ---
    if not candidates:
        # Hard Rule 6 guard: log the reason
        _log_not_recommended(request_id, user_methods, amount_safe_today, requested_amount)
        return Recommendation(
            request_id=request_id,
            amount_safe_to_pay=round(amount_safe_today, 2),
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan="none",
            earliest_date_for_full_payment="",
            spending_changes_needed="none",
            decision_explanation="",
        )

    best = candidates[0]
    affordability = _derive_affordability(best, request_date, amount_safe_today, requested_amount)

    # Hard Rule 5: wait → payment_plan must be "none"
    payment_plan_out = best.payment_plan_str
    if best.method == "wait":
        payment_plan_out = "none"

    spending_str = best.spending_changes.to_string() if best.spending_changes else "none"

    logger.info(
        "Ranked recommendation for %s: method=%s, status=%s, total=%.2f, "
        "completes_by_deadline=%s, spending=%s",
        request_id, best.method, affordability, best.total_paid,
        best.completes_by_deadline, spending_str
    )

    return Recommendation(
        request_id=request_id,
        amount_safe_to_pay=round(amount_safe_today, 2),
        affordability_status=affordability,
        recommended_payment_method=best.method,
        payment_plan=payment_plan_out,
        earliest_date_for_full_payment=earliest_full_str,
        spending_changes_needed=spending_str,
        decision_explanation="",  # Phase 9
    )


def _derive_affordability(
    best: CandidatePlan,
    request_date: pd.Timestamp,
    amount_safe_today: float,
    requested_amount: float,
) -> str:
    """
    Derive affordability_status from the winning plan.
      affordable_now          → full_payment, no spending changes, today
      affordable_with_plan    → full amount completed via partial/installments/spending changes
      affordable_later        → wait (full payment on a future date)
      not_affordable          → not_recommended
    """
    if best.method == "not_recommended":
        return "not_affordable"
    if best.method == "wait":
        return "affordable_later"
    if best.method == "full_payment" and best.spending_changes is None and best.first_payment_date == request_date:
        return "affordable_now"
    if best.method in ("full_payment", "partial_payment", "installments"):
        return "affordable_with_plan"
    return "not_affordable"


def _log_not_recommended(
    request_id: str,
    user_methods: set,
    amount_safe_today: float,
    requested_amount: float,
) -> None:
    """Hard Rule 6: log why not_recommended was chosen."""
    logger.info(
        "not_recommended for %s: user_methods=%s, amount_safe=%.2f, requested=%.2f. "
        "full_payment_in_methods=%s",
        request_id,
        user_methods,
        amount_safe_today,
        requested_amount,
        "full_payment" in user_methods,
    )
