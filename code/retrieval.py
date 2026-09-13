"""
Phase 2 — Retrieval.
Hard ID-filter joins for user/request/event data.
Returns a structured context dict for a single request.
"""
from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd

from loaders import (
    load_profiles,
    load_events,
    load_payment_options,
    load_messages,
    load_images,
    load_sample_requests,
    load_requests,
    resolve_event_amounts,
    MEDIA_DIR,
)


def get_request_context(
    request_id: str,
    use_sample: bool = False,
) -> Dict[str, Any]:
    """
    Build all context for a single request_id. Returns a dict with:
      request       — Series (the request row)
      profile       — Series (user financial profile)
      events        — DataFrame (all events for the user, amounts in home currency)
      payment_opts  — DataFrame (payment options for this request)
      messages      — DataFrame (messages for this user or request)
      images        — DataFrame (image links for this request, with file_path)
      home_currency — str
    """
    # Load base tables
    req_df = load_sample_requests() if use_sample else load_requests()
    profiles = load_profiles()
    all_events = load_events()
    all_options = load_payment_options()
    all_messages = load_messages()
    all_images = load_images()

    # Locate request
    if request_id not in req_df.index:
        raise KeyError(f"request_id {request_id!r} not found")
    req = req_df.loc[request_id]
    user_id = req["user_id"]

    # Profile
    if user_id not in profiles.index:
        raise KeyError(f"user_id {user_id!r} not found in profiles")
    profile = profiles.loc[user_id]
    home_currency = profile["home_currency"]

    # Events — filter by user, resolve amounts to home currency
    user_events = all_events[all_events["user_id"] == user_id].copy()
    user_events = resolve_event_amounts(user_events, home_currency)

    # Payment options — filter by request
    req_options = all_options[all_options["request_id"] == request_id].copy()

    # Messages — for this user or request (ordered by sent_at)
    msg_mask = (all_messages["user_id"] == user_id) | (all_messages["request_id"] == request_id)
    req_messages = all_messages[msg_mask].copy().sort_values("sent_at")

    # Images — for this request
    img_mask = (all_images["request_id"] == request_id)
    req_images = all_images[img_mask].copy()
    # Add file paths
    req_images = req_images.copy()
    req_images["file_path"] = req_images["image_id"].apply(
        lambda iid: str(MEDIA_DIR / f"{iid}.png")
    )

    return {
        "request_id": request_id,
        "request": req,
        "profile": profile,
        "events": user_events,
        "payment_opts": req_options,
        "messages": req_messages,
        "images": req_images,
        "home_currency": home_currency,
    }
