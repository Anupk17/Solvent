"""
Phase 4 — 90-day forecast engine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from classifier import EventClassifier, EXCLUDE_STATUSES

logger = logging.getLogger(__name__)

FORECAST_DAYS = 90


@dataclass
class LedgerEntry:
    date: pd.Timestamp
    category: str
    direction: str
    amount: float
    balance_after: float
    source: str
    event_id: str = ""
    flexibility: str = "fixed"


@dataclass
class ForecastResult:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    starting_balance: float
    minimum_balance: float
    ledger: List[LedgerEntry] = field(default_factory=list)
    min_balance_seen: float = float("inf")
    total_projected_income: float = 0.0
    total_projected_expenses: float = 0.0

    def balance_on(self, target_date: pd.Timestamp) -> float:
        balance = self.starting_balance
        for entry in self.ledger:
            if entry.date <= target_date:
                balance = entry.balance_after
            else:
                break
        return balance

    def is_safe(self) -> bool:
        return self.min_balance_seen >= self.minimum_balance

    def min_safe_balance_on_or_after(self, target_date: pd.Timestamp) -> float:
        balances = [e.balance_after for e in self.ledger if e.date >= target_date]
        if not balances:
            return self.balance_on(target_date)
        return min(balances)


class ForecastEngine:
    def __init__(
        self,
        classifier: EventClassifier,
        profile: pd.Series,
        starting_balance: float,
        request_date: pd.Timestamp,
        spending_overrides: Optional[Dict[str, Optional[float]]] = None,
        income_override: Optional[Dict] = None,
    ):
        self.classifier = classifier
        self.profile = profile
        self.starting_balance = starting_balance
        self.request_date = request_date
        self.minimum_balance = float(profile["minimum_balance_to_keep"])
        self.spending_overrides = spending_overrides or {}
        self.income_override = income_override

    def run(
        self,
        extra_payments: Optional[List[Tuple[pd.Timestamp, float]]] = None,
        end_date_override: Optional[pd.Timestamp] = None,
    ) -> ForecastResult:
        extra_payments = extra_payments or []
        end_date = end_date_override if end_date_override else (
            self.request_date + pd.Timedelta(days=FORECAST_DAYS)
        )

        events = self._collect_events(end_date, extra_payments)

        # Sort: by date, then credits before debits, then confirmed before projected
        def sort_key(e):
            direction_order = 0 if e["direction"] == "credit" else 1
            source_order = 0 if e["source"] == "confirmed" else 1
            return (e["date"], direction_order, source_order)
        events.sort(key=sort_key)

        balance = self.starting_balance
        ledger: List[LedgerEntry] = []
        min_seen = balance
        total_income = 0.0
        total_expenses = 0.0

        for ev in events:
            d = ev["date"]
            if ev["direction"] == "credit":
                balance += ev["amount"]
                total_income += ev["amount"]
            else:
                balance -= ev["amount"]
                total_expenses += ev["amount"]

            entry = LedgerEntry(
                date=d,
                category=ev["category"],
                direction=ev["direction"],
                amount=ev["amount"],
                balance_after=balance,
                source=ev["source"],
                event_id=ev.get("event_id", ""),
                flexibility=ev.get("flexibility", "fixed"),
            )
            ledger.append(entry)
            if balance < min_seen:
                min_seen = balance

        result = ForecastResult(
            start_date=self.request_date,
            end_date=end_date,
            starting_balance=self.starting_balance,
            minimum_balance=self.minimum_balance,
            ledger=ledger,
            min_balance_seen=min_seen,
            total_projected_income=total_income,
            total_projected_expenses=total_expenses,
        )

        self.classifier.sanity_check_income(total_income)
        self._log_forecast(result)
        return result

    def _collect_events(self, end_date, extra_payments):
        events = []

        confirmed = self.classifier.get_confirmed_future_events()
        confirmed_categories_by_date: Dict[str, List[pd.Timestamp]] = {}

        for _, row in confirmed.iterrows():
            d = row["settlement_date"]
            if pd.isna(d) or d > end_date or d < self.request_date:
                continue
            amt = row["amount_home"]
            if pd.isna(amt):
                continue
            eid = row["event_id"]
            if eid in self.spending_overrides:
                override_val = self.spending_overrides[eid]
                if override_val is None:
                    continue
                amt = override_val
            cat = row["category"]
            direction = row["direction"]
            events.append({
                "date": d,
                "category": cat,
                "direction": direction,
                "amount": float(amt),
                "source": "confirmed",
                "event_id": eid,
                "flexibility": row["flexibility"],
            })
            key = f"{cat}__{direction}"
            if key not in confirmed_categories_by_date:
                confirmed_categories_by_date[key] = []
            confirmed_categories_by_date[key].append(d)

        projections = self.classifier.project_recurring_events(
            self.request_date, end_date, income_override=self.income_override
        )
        for proj in projections:
            d = proj["date"]
            cat = proj["category"]
            direction = proj["direction"]
            key = f"{cat}__{direction}"
            amt = proj["amount_home"]

            cat_override = self._get_category_override(cat, direction)
            if cat_override is not None:
                if cat_override == 0.0:
                    continue
                amt = cat_override

            if key in confirmed_categories_by_date:
                conf_dates = confirmed_categories_by_date[key]
                freq = 28
                if any(abs((d - cd).days) <= freq // 2 for cd in conf_dates):
                    continue

            events.append({
                "date": d,
                "category": cat,
                "direction": direction,
                "amount": float(amt),
                "source": "projected",
                "event_id": proj["event_id"],
                "flexibility": proj["flexibility"],
            })

        for pay_date, pay_amount in extra_payments:
            if self.request_date <= pay_date <= end_date:
                events.append({
                    "date": pay_date,
                    "category": "payment_plan",
                    "direction": "debit",
                    "amount": float(pay_amount),
                    "source": "payment_plan",
                    "event_id": "plan_payment",
                    "flexibility": "fixed",
                })

        return events

    def _get_category_override(self, category, direction):
        for eid, new_val in self.spending_overrides.items():
            evs = self.classifier.events[self.classifier.events["event_id"] == eid]
            if len(evs) == 0:
                continue
            row = evs.iloc[0]
            if row["category"] == category and row["direction"] == direction:
                return 0.0 if new_val is None else float(new_val)
        return None

    def _log_forecast(self, result):
        logger.info(
            "Forecast [%s to %s]: start_bal=%.2f, min_bal=%.2f, "
            "min_seen=%.2f, income=%.2f, expenses=%.2f, safe=%s",
            result.start_date.date(), result.end_date.date(),
            result.starting_balance, result.minimum_balance,
            result.min_balance_seen, result.total_projected_income,
            result.total_projected_expenses, result.is_safe(),
        )


def is_payment_safe(
    engine,
    payment_date,
    payment_amount,
    additional_payments=None,
):
    all_payments = list(additional_payments or [])
    all_payments.append((payment_date, payment_amount))
    result = engine.run(extra_payments=all_payments)
    return result.is_safe()


def is_wait_safe(
    engine: ForecastEngine,
    payment_date: pd.Timestamp,
    payment_amount: float,
) -> bool:
    """
    Like is_payment_safe but extends the forecast window to payment_date + 45 days.
    Used specifically for find_earliest_date_for_full_payment (wait scenario).
    This captures the next salary cycle after the payment date, preventing
    false negatives where the balance recovers from the next paycheck just
    outside the standard 90-day window.
    """
    all_payments = [(payment_date, payment_amount)]
    extended_end = payment_date + pd.Timedelta(days=45)
    result = engine.run(extra_payments=all_payments, end_date_override=extended_end)
    return result.is_safe()
