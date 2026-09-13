"""
Phase 8 orchestration.

For each request, extract:
  - Image amounts (for blank-amount events linked to request images)
  - Message amendments (salary changes, rent changes, etc.)

Returns an 'amendments' dict ready to pass to pipeline.process_request().
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from extraction import extract_from_image, extract_from_message, MessageAmendment

logger = logging.getLogger(__name__)


def build_amendments(
    user_id: str,
    request_id: str,
    request_date: pd.Timestamp,
    events: pd.DataFrame,
    messages: pd.DataFrame,
    images: pd.DataFrame,
    home_currency: str,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    Run all extraction for a request and return consolidated amendments dict.
    """
    amendments: Dict[str, Any] = {}

    if not use_llm:
        return amendments

    # -----------------------------------------------------------------------
    # 1. Image extraction — only for events with blank amounts
    # -----------------------------------------------------------------------
    blank_event_ids = set(events[events["amount"].isna()]["event_id"].tolist())

    image_amounts: Dict[str, float] = {}
    for _, img_row in images.iterrows():
        related_eid = img_row["related_event_id"]
        if not related_eid:
            continue
        if related_eid not in blank_event_ids:
            continue  # Only extract when amount is actually needed
        image_id = img_row["image_id"]
        file_path = img_row["file_path"]
        logger.info("Extracting amount from image %s for event %s", image_id, related_eid)
        result = extract_from_image(image_id, file_path)
        if result.amount is not None:
            # Convert to home currency if needed
            if result.currency and result.currency != home_currency:
                try:
                    from loaders import convert_to_home_currency
                    # Use a reasonable settlement date (request_date as fallback)
                    event_rows = events[events["event_id"] == related_eid]
                    if len(event_rows) > 0 and pd.notna(event_rows.iloc[0]["settlement_date"]):
                        sdate = event_rows.iloc[0]["settlement_date"]
                    else:
                        sdate = request_date
                    converted = convert_to_home_currency(
                        result.amount, result.currency, home_currency, sdate
                    )
                    image_amounts[related_eid] = converted
                    logger.info(
                        "Converted image amount %.2f %s -> %.2f %s for event %s",
                        result.amount, result.currency, converted, home_currency, related_eid
                    )
                except Exception as e:
                    logger.warning("Currency conversion failed for image %s: %s", image_id, e)
                    image_amounts[related_eid] = result.amount
            else:
                image_amounts[related_eid] = result.amount

    if image_amounts:
        amendments["image_amounts"] = image_amounts

    # -----------------------------------------------------------------------
    # 2. Message extraction
    # -----------------------------------------------------------------------
    for _, msg_row in messages.iterrows():
        msg_id = str(msg_row["message_id"])
        msg_text = str(msg_row["message_text"])
        related_eid = str(msg_row.get("related_event_id", ""))
        sent_at = msg_row.get("sent_at")

        # Only process messages sent before or on request_date
        if pd.notna(sent_at):
            msg_date = pd.Timestamp(sent_at)
            if msg_date.tz_localize(None) if msg_date.tzinfo else msg_date > request_date:
                logger.debug("Skipping future message %s (sent %s > request %s)", msg_id, msg_date.date(), request_date.date())
                continue

        amend = extract_from_message(msg_id, msg_text, related_event_id=related_eid)
        _apply_message_amendment(amendments, amend, events, request_date)

    return amendments


def _apply_message_amendment(
    amendments: Dict[str, Any],
    amend: MessageAmendment,
    events: pd.DataFrame,
    request_date: pd.Timestamp,
) -> None:
    """
    Merge a message amendment into the amendments dict.
    Most recent amendment for a given type wins (messages are processed
    in sent_at order by the caller).
    """
    action = amend.action

    if action == "none":
        return

    elif action == "salary_change":
        if amend.new_amount is not None:
            existing = amendments.get("salary_override", {})
            # Only update if this message is more recent (caller processes in order)
            amendments["salary_override"] = {
                "amount": amend.new_amount,
                "date": amend.new_date,
                "currency": amend.currency,
            }
            logger.info("Salary override: %.2f %s from %s", amend.new_amount, amend.currency, amend.new_date)

    elif action == "salary_date_change":
        if amend.new_date:
            amendments["salary_date_change"] = {"date": amend.new_date}
            logger.info("Salary date change to %s", amend.new_date)

    elif action == "rent_change":
        # Check for percentage increase in the message notes
        import re
        pct_match = re.search(r"(\d+)\s*%", amend.notes or "")
        if pct_match:
            pct = float(pct_match.group(1))
            amendments["rent_change"] = {"pct_increase": pct}
            logger.info("Rent increase %.1f%% applied", pct)
        elif amend.new_amount and amend.related_event_id:
            amendments["rent_change"] = {
                "event_id": amend.related_event_id,
                "new_amount": amend.new_amount,
            }

    elif action == "income_ended":
        # Mark future salary events as cancelled
        if "income_ended_category" not in amendments:
            amendments["income_ended_category"] = "salary"
            logger.info("Income ended — marking future salary as cancelled")

    elif action in ("refund_pending",):
        # Pending refund — NOT countable as income per §6.3
        # The related event should already be marked pending in events
        # No amendment needed — forecast already excludes pending credits
        logger.debug("Refund pending for msg %s — no amendment needed", amend.message_id)

    elif action in ("refund_settled",):
        # Refund has settled — it's already in events as settled credit
        # No amendment needed
        logger.debug("Refund settled msg %s — already in events", amend.message_id)

    elif action in ("investment_valuation",):
        # Non-cash — never add to forecast
        logger.debug("Investment valuation msg %s — non-cash, skipped", amend.message_id)

    elif action in ("bonus_pending", "bonus_unconfirmed"):
        # Per §6.3: do not count pending bonuses
        logger.debug("Bonus pending/unconfirmed msg %s — excluded from income", amend.message_id)

    elif action == "event_confirmed":
        # A previously uncertain event is now confirmed
        if amend.related_event_id:
            logger.debug("Event confirmed: %s", amend.related_event_id)
