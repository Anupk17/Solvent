"""
Phase 1 — Loaders for all 9 CSVs.
All paths are resolved relative to the repo root (two levels up from this file).
Currency conversion uses settlement_date for rate lookup.
"""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Repo / dataset root
# ---------------------------------------------------------------------------
_CODE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = _CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"
MEDIA_DIR = DATASET_DIR / "media" / "images"


# ---------------------------------------------------------------------------
# Raw loaders (called once, results cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_profiles() -> pd.DataFrame:
    df = pd.read_csv(DATASET_DIR / "financial_profiles.csv")
    # Parse pipe-separated list fields into Python lists
    list_fields = [
        "financial_priorities",
        "expense_categories_to_protect",
        "expense_categories_user_is_willing_to_reduce",
        "expense_categories_user_is_willing_to_stop",
        "payment_methods_user_will_consider",
    ]
    for col in list_fields:
        df[col] = df[col].fillna("").apply(
            lambda x: [v.strip() for v in x.split("|") if v.strip()] if x else []
        )
    df["max_installment_months"] = pd.to_numeric(df["max_installment_months"], errors="coerce")
    return df.set_index("user_id")


@lru_cache(maxsize=1)
def load_events() -> pd.DataFrame:
    df = pd.read_csv(
        DATASET_DIR / "financial_events.csv",
        parse_dates=["event_date", "settlement_date"],
    )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")  # NaN for blanks
    df["minimum_allowed_amount"] = pd.to_numeric(df["minimum_allowed_amount"], errors="coerce")
    df["linked_event_id"] = df["linked_event_id"].fillna("")
    df["flexibility"] = df["flexibility"].fillna("fixed")
    df["currency"] = df["currency"].fillna("")
    return df


@lru_cache(maxsize=1)
def load_exchange_rates() -> pd.DataFrame:
    df = pd.read_csv(DATASET_DIR / "exchange_rates.csv", parse_dates=["rate_date"])
    return df


@lru_cache(maxsize=1)
def load_requests() -> pd.DataFrame:
    df = pd.read_csv(DATASET_DIR / "requests.csv", parse_dates=["request_date", "desired_completion_date"])
    df["allows_partial_payment"] = df["allows_partial_payment"].astype(str).str.lower() == "true"
    return df.set_index("request_id")


@lru_cache(maxsize=1)
def load_sample_requests() -> pd.DataFrame:
    df = pd.read_csv(
        DATASET_DIR / "sample_requests.csv",
        parse_dates=["request_date", "desired_completion_date"],
    )
    df["allows_partial_payment"] = df["allows_partial_payment"].astype(str).str.lower() == "true"
    return df.set_index("request_id")


@lru_cache(maxsize=1)
def load_payment_options() -> pd.DataFrame:
    df = pd.read_csv(DATASET_DIR / "request_payment_options.csv", parse_dates=["first_payment_date"])
    df["payment_frequency_days"] = pd.to_numeric(df["payment_frequency_days"], errors="coerce")
    df["financing_fee"] = pd.to_numeric(df["financing_fee"], errors="coerce").fillna(0.0)
    return df


@lru_cache(maxsize=1)
def load_messages() -> pd.DataFrame:
    df = pd.read_csv(DATASET_DIR / "messages.csv", parse_dates=["sent_at"])
    df["related_event_id"] = df["related_event_id"].fillna("")
    df["request_id"] = df["request_id"].fillna("")
    return df


@lru_cache(maxsize=1)
def load_images() -> pd.DataFrame:
    df = pd.read_csv(DATASET_DIR / "images.csv")
    df["related_event_id"] = df["related_event_id"].fillna("")
    df["request_id"] = df["request_id"].fillna("")
    return df


# ---------------------------------------------------------------------------
# Currency conversion
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _build_rate_index() -> dict:
    """
    Build a nested lookup: (from_currency, to_currency) -> sorted list of (date, rate).
    Rates are stored as (rate_date, rate) sorted ascending.
    """
    er = load_exchange_rates()
    index: dict = {}
    for _, row in er.iterrows():
        key = (row["from_currency"], row["to_currency"])
        if key not in index:
            index[key] = []
        index[key].append((row["rate_date"], row["rate"]))
    # Sort by date
    for key in index:
        index[key].sort(key=lambda x: x[0])
    return index


def convert_to_home_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
    settlement_date: pd.Timestamp,
) -> float:
    """
    Convert amount from from_currency to to_currency using the rate whose
    rate_date is closest on or before settlement_date (or nearest available).

    If from_currency == to_currency, returns amount unchanged.
    Supports chained conversion (e.g. USD -> IDR via EUR -> USD intermediate)
    is NOT needed — all rates in the table are direct.
    """
    if from_currency == to_currency:
        return amount

    rates = _build_rate_index()
    key = (from_currency, to_currency)

    # Try direct rate
    rate = _find_rate(rates, key, settlement_date)
    if rate is not None:
        return amount * rate

    # Try inverse
    inv_key = (to_currency, from_currency)
    inv_rate = _find_rate(rates, inv_key, settlement_date)
    if inv_rate is not None and inv_rate != 0:
        return amount / inv_rate

    # Try chain: from_currency -> USD -> to_currency
    chain_currencies = ["USD", "EUR"]
    for mid in chain_currencies:
        if mid == from_currency or mid == to_currency:
            continue
        r1 = _find_rate(rates, (from_currency, mid), settlement_date)
        if r1 is None:
            r1_inv = _find_rate(rates, (mid, from_currency), settlement_date)
            r1 = (1.0 / r1_inv) if r1_inv else None
        r2 = _find_rate(rates, (mid, to_currency), settlement_date)
        if r2 is None:
            r2_inv = _find_rate(rates, (to_currency, mid), settlement_date)
            r2 = (1.0 / r2_inv) if r2_inv else None
        if r1 is not None and r2 is not None:
            return amount * r1 * r2

    # Fallback: return as-is (shouldn't happen with provided rates)
    import warnings
    warnings.warn(
        f"No exchange rate found for {from_currency}->{to_currency} on {settlement_date}. "
        f"Returning unconverted amount."
    )
    return amount


def _find_rate(
    rates: dict,
    key: tuple,
    settlement_date: pd.Timestamp,
) -> Optional[float]:
    """Return the rate for key whose rate_date <= settlement_date (closest)."""
    if key not in rates:
        return None
    pairs = rates[key]
    # Find the latest rate_date <= settlement_date
    best = None
    for date, rate in pairs:
        if date <= settlement_date:
            best = rate
        else:
            break
    if best is not None:
        return best
    # All dates are after settlement_date — use the earliest available
    return pairs[0][1]


# ---------------------------------------------------------------------------
# Event amount resolution (converts to home_currency in place)
# ---------------------------------------------------------------------------

def resolve_event_amounts(events: pd.DataFrame, home_currency: str) -> pd.DataFrame:
    """
    Return a copy of events with `amount_home` column = amount in home_currency.
    Events with blank (NaN) amounts will have NaN in amount_home too.
    """
    events = events.copy()
    amounts_home = []
    for _, row in events.iterrows():
        amt = row["amount"]
        if pd.isna(amt):
            amounts_home.append(float("nan"))
            continue
        curr = row["currency"]
        sdate = row["settlement_date"]
        if pd.isna(sdate):
            sdate = row["event_date"]
        if pd.isna(sdate):
            sdate = pd.Timestamp("2025-01-15")  # fallback
        converted = convert_to_home_currency(float(amt), str(curr), home_currency, sdate)
        amounts_home.append(converted)
    events["amount_home"] = amounts_home
    return events
