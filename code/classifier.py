"""
Phase 3 — Event classification.

Identifies recurring vs one-time events and flexible vs essential spending.
Classification is ALWAYS by explicit fields (category, flexibility, linked_event_id,
event_type, status) — NEVER by description string matching.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_RECURRENCE_COUNT = 2
MIN_RECURRENCE_COUNT_INCOME = 1

CASH_STATUSES = {"settled", "scheduled", "pending"}
EXCLUDE_STATUSES = {"failed", "cancelled", "unrealized"}


def _is_cash_event(row: pd.Series) -> bool:
    if row["status"] in EXCLUDE_STATUSES:
        return False
    if row["event_type"] == "investment_valuation":
        return False
    if row["direction"] == "non_cash":
        return False
    return True


def _is_dedup_linked(row: pd.Series) -> bool:
    if not row["linked_event_id"]:
        return False
    if row["event_type"] in ("refund", "investment_valuation"):
        return True
    return False


class EventClassifier:
    def __init__(self, events: pd.DataFrame, profile: pd.Series, request_date: pd.Timestamp):
        self.events = events.copy()
        self.profile = profile
        self.request_date = request_date
        self.home_currency = profile["home_currency"]
        self.protected_categories: set = set(profile.get("expense_categories_to_protect", []))
        self.reducible_categories: set = set(profile.get("expense_categories_user_is_willing_to_reduce", []))
        self.stoppable_categories: set = set(profile.get("expense_categories_user_is_willing_to_stop", []))
        self._settled_history: Optional[pd.DataFrame] = None
        self._recurring_patterns: Optional[Dict] = None
        # These can be set by pipeline.py
        self._income_ended_categories: set = set()
        self._suppressed_debit_categories: set = set()
        self._salary_next_date: Optional[pd.Timestamp] = None
        self._suppress_projected_expenses: bool = False  # if True, no projected expense patterns

    def get_active_events(self) -> pd.DataFrame:
        ev = self.events
        mask = ev.apply(lambda r: _is_cash_event(r) and not _is_dedup_linked(r), axis=1)
        return ev[mask].copy()

    def get_settled_history(self) -> pd.DataFrame:
        """Settled events + scheduled credits (for income pattern detection)."""
        if self._settled_history is None:
            ev = self.get_active_events()
            self._settled_history = ev[
                (ev["status"] == "settled") |
                ((ev["status"] == "scheduled") & (ev["direction"] == "credit"))
            ].copy()
        return self._settled_history

    def get_recurring_patterns(self) -> Dict[str, dict]:
        if self._recurring_patterns is not None:
            return self._recurring_patterns

        history = self.get_settled_history()
        patterns: Dict[str, dict] = {}

        for cat, grp in history.groupby("category"):
            for direction, dgrp in grp.groupby("direction"):
                min_count = MIN_RECURRENCE_COUNT_INCOME if direction == "credit" else MIN_RECURRENCE_COUNT
                if len(dgrp) < min_count:
                    continue
                if direction == "non_cash":
                    continue

                dgrp_sorted = dgrp.sort_values("settlement_date")
                amounts = dgrp_sorted["amount_home"].dropna().values
                if len(amounts) == 0:
                    continue

                typical_amount = float(np.median(amounts))
                days_of_month = [d.day for d in dgrp_sorted["settlement_date"] if pd.notna(d)]
                day_of_month = int(np.median(days_of_month)) if days_of_month else 1

                dates = [d for d in dgrp_sorted["settlement_date"] if pd.notna(d)]
                if len(dates) >= 2:
                    gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                    reasonable_gaps = [g for g in gaps if 1 <= g <= 60]
                    if reasonable_gaps:
                        gap_counts = Counter(reasonable_gaps)
                        weekly = sum(v for k, v in gap_counts.items() if 1 <= k <= 10)
                        biweekly = sum(v for k, v in gap_counts.items() if 11 <= k <= 20)
                        monthly = sum(v for k, v in gap_counts.items() if 21 <= k <= 60)
                        if weekly >= max(biweekly, monthly):
                            freq = int(np.median([g for g in reasonable_gaps if 1 <= g <= 10]))
                        elif biweekly >= monthly:
                            freq = int(np.median([g for g in reasonable_gaps if 11 <= g <= 20]))
                        else:
                            freq = int(np.median([g for g in reasonable_gaps if 21 <= g <= 60]))
                    else:
                        freq = 30
                    # For salary with double-payment (15th + 20th), force monthly
                    if cat == "salary" and direction == "credit" and freq < 20:
                        monthly_gaps = [g for g in gaps if 20 <= g <= 60]
                        if monthly_gaps:
                            freq = int(np.median(monthly_gaps))
                else:
                    freq = 30

                flex_vals = dgrp["flexibility"].value_counts()
                flexibility = flex_vals.index[0] if len(flex_vals) > 0 else "fixed"

                # Detect "ended" patterns:
                # A debit category is skipped if:
                #   1. Its expected next occurrence already passed (overdue by ≥1 day), AND
                #   2. It looks like a fixed-term series (≤6 occurrences, identical amounts), AND
                #   3. No future scheduled/pending event exists for this category
                last_settle = dgrp_sorted["settlement_date"].max()
                if pd.notna(last_settle) and direction == "debit":
                    expected_next = last_settle + pd.Timedelta(days=freq)
                    overdue = expected_next < self.request_date

                    if overdue:
                        # Only suppress fixed-term series (small count, identical amounts)
                        all_same = len(set(np.round(amounts, 2))) == 1
                        is_fixed = all_same and len(dgrp) <= 6
                        if is_fixed:
                            active = self.get_active_events()
                            future_sched = active[
                                (active["category"] == cat) &
                                (active["direction"] == "debit") &
                                (active["status"].isin({"scheduled", "pending"})) &
                                (active["settlement_date"] >= self.request_date)
                            ]
                            if len(future_sched) == 0:
                                logger.debug(
                                    "Skipping ended fixed-term debit %s: last=%s expected_next=%s",
                                    cat, last_settle.date(), expected_next.date()
                                )
                                continue

                key = f"{cat}__{direction}"
                patterns[key] = {
                    "category": cat,
                    "direction": direction,
                    "typical_amount": typical_amount,
                    "day_of_month": day_of_month,
                    "frequency_days": freq,
                    "flexibility": flexibility,
                    "count": len(dgrp),
                    "example_event_ids": list(dgrp_sorted["event_id"].head(3)),
                }

        self._recurring_patterns = patterns
        return patterns

    def get_confirmed_future_events(self) -> pd.DataFrame:
        ev = self.get_active_events()
        future = ev[
            (ev["status"].isin({"scheduled", "pending"})) &
            (ev["settlement_date"] >= self.request_date)
        ].copy()
        # Exclude pending credits except confirmed scheduled salary
        pending_credits_excl = (
            (future["status"] == "pending") &
            (future["direction"] == "credit") &
            (future["category"] != "salary")
        )
        future = future[~pending_credits_excl]
        return future

    def project_recurring_events(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        income_override: Optional[Dict] = None,
    ) -> List[dict]:
        """
        Generate projected recurring events.
        Income categories in _income_ended_categories are suppressed.
        income_override allows amending salary amount from a message.
        """
        patterns = self.get_recurring_patterns()
        projections: List[dict] = []

        confirmed = self.get_confirmed_future_events()
        confirmed_income_cats = set(
            confirmed[confirmed["direction"] == "credit"]["category"].tolist()
        )

        for key, pat in patterns.items():
            cat = pat["category"]
            direction = pat["direction"]
            freq = pat["frequency_days"]
            amount = pat["typical_amount"]
            flexibility = pat["flexibility"]
            day_of_month = pat["day_of_month"]

            # Skip explicitly ended income categories
            if direction == "credit" and cat in self._income_ended_categories:
                logger.debug("Skipping income projection for %s (income_ended)", cat)
                continue

            # Skip explicitly suppressed debit categories (from amendments)
            if direction == "debit" and cat in self._suppressed_debit_categories:
                logger.debug("Skipping suppressed debit %s", cat)
                continue

            # When suppress_projected_expenses is set, skip ALL projected debit patterns
            if direction == "debit" and self._suppress_projected_expenses:
                continue

            # Apply income override (salary amendment from message)
            if direction == "credit" and income_override and income_override.get("category", cat) == cat:
                override_amount = income_override.get("amount", amount)
                override_start = income_override.get("start_date", start_date)
                amount = override_amount if override_start <= end_date else amount

            # Find last known settlement date for this category+direction
            history = self.get_settled_history()
            cat_hist = history[
                (history["category"] == cat) &
                (history["direction"] == direction)
            ].sort_values("settlement_date")

            last_date = cat_hist["settlement_date"].max() if len(cat_hist) > 0 else start_date
            if pd.isna(last_date):
                last_date = start_date

            # Generate next occurrences
            # For salary with date change, use the overridden next date
            if cat == "salary" and direction == "credit" and self._salary_next_date is not None:
                next_date = self._salary_next_date
            else:
                next_date = last_date + timedelta(days=freq)
                if 25 <= freq <= 35:
                    try:
                        next_date = _next_monthly_date(last_date, day_of_month)
                    except Exception:
                        pass

            confirmed_cats_by_date: Dict[str, List[pd.Timestamp]] = {}
            key2 = f"{cat}__{direction}"
            conf_in_cat = confirmed[(confirmed["category"] == cat) & (confirmed["direction"] == direction)]
            if len(conf_in_cat) > 0:
                confirmed_cats_by_date[key2] = list(conf_in_cat["settlement_date"])

            count = 0
            d = next_date
            while d <= end_date:
                if d >= start_date:
                    # Skip if confirmed event already covers this date
                    if key2 in confirmed_cats_by_date:
                        conf_dates = confirmed_cats_by_date[key2]
                        if any(abs((d - cd).days) <= max(freq // 2, 7) for cd in conf_dates):
                            if 25 <= freq <= 35:
                                d = _next_monthly_date(d, day_of_month)
                            else:
                                d = d + timedelta(days=freq)
                            count += 1
                            continue

                    projections.append({
                        "date": d,
                        "category": cat,
                        "direction": direction,
                        "amount_home": amount,
                        "flexibility": flexibility,
                        "source": "projected",
                        "event_id": f"proj_{key}_{count}",
                    })
                count += 1
                if 25 <= freq <= 35:
                    try:
                        d = _next_monthly_date(d, day_of_month)
                    except Exception:
                        d = d + timedelta(days=freq)
                else:
                    d = d + timedelta(days=freq)
                if count > 200:
                    break

        return projections

    def is_flexible(self, event_id: str) -> bool:
        ev = self.events
        row = ev[ev["event_id"] == event_id]
        if len(row) == 0:
            return False
        flex = row.iloc[0]["flexibility"]
        cat = row.iloc[0]["category"]
        if flex not in ("reducible", "stoppable", "reducible_or_stoppable"):
            return False
        if cat in self.protected_categories:
            return False
        return True

    def sanity_check_income(self, forecast_income: float, forecast_days: int = 90) -> None:
        history = self.get_settled_history()
        ninety_ago = self.request_date - pd.Timedelta(days=90)
        recent_income = history[
            (history["direction"] == "credit") &
            (history["settlement_date"] >= ninety_ago) &
            (history["settlement_date"] < self.request_date) &
            (history["category"] == "salary")
        ]["amount_home"].sum()
        if recent_income > 0 and forecast_income < recent_income * 0.3:
            logger.warning(
                "HARD RULE 3 GUARD: Forecast income %.2f is less than 30%% of "
                "recent 90-day settled income %.2f. Check recurring income projection.",
                forecast_income, recent_income,
            )


def _next_monthly_date(from_date: pd.Timestamp, day_of_month: int) -> pd.Timestamp:
    from calendar import monthrange
    year = from_date.year
    month = from_date.month + 1
    if month > 12:
        month = 1
        year += 1
    max_day = monthrange(year, month)[1]
    actual_day = min(day_of_month, max_day)
    return pd.Timestamp(year=year, month=month, day=actual_day)
