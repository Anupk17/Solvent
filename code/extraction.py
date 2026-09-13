"""
Phase 8 — Extraction.

Hard Rule 2: Tesseract OCR first → text LLM to parse → vision LLM fallback.
Hard Rule 4: Cache has explicit status "ok"/"failed". Failed entries are NEVER
             treated as resolved. They get retried on next run.

Hard Rule 1: Extraction extracts FACTS only. It never computes financial figures.

This module handles:
  A. Image extraction: amount + currency from images (for blank-amount events)
  B. Message parsing: structured amendments (salary changes, rent changes, etc.)
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.resolve()
CACHE_DIR = REPO_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)
IMAGE_CACHE_FILE = CACHE_DIR / "image_cache.json"
MSG_CACHE_FILE = CACHE_DIR / "message_cache.json"


# ---------------------------------------------------------------------------
# Cache helpers (Hard Rule 4 — status must be explicit)
# ---------------------------------------------------------------------------

def _load_cache(path: Path) -> Dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(path: Path, data: Dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("Cache write error: %s", e)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ImageExtraction:
    image_id: str
    amount: Optional[float]
    currency: Optional[str]
    date: Optional[str]
    notes: str
    source: str   # "ocr_llm" | "vision_llm" | "unresolvable"
    raw_ocr: str = ""


@dataclass
class MessageAmendment:
    message_id: str
    action: str            # "none" | "salary_change" | "rent_change" | "event_cancel" | "event_confirmed"
    new_amount: Optional[float] = None
    new_date: Optional[str] = None
    currency: Optional[str] = None
    related_event_id: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Tesseract OCR
# ---------------------------------------------------------------------------

def run_ocr(image_path: str) -> str:
    """Run Tesseract OCR on an image, return raw text (may be empty)."""
    try:
        import pytesseract
        from PIL import Image

        # Try to set tesseract path if configured
        tess_cmd = os.environ.get("TESSERACT_CMD")
        if tess_cmd:
            pytesseract.pytesseract.tesseract_cmd = tess_cmd

        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, config="--psm 6")
        return text.strip()
    except Exception as e:
        logger.warning("OCR failed for %s: %s", image_path, e)
        return ""


# ---------------------------------------------------------------------------
# LLM client setup (lazy init)
# ---------------------------------------------------------------------------

_llm_client = None
_vision_client = None


def _get_groq_client():
    global _llm_client
    if _llm_client is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        _llm_client = Groq(api_key=api_key)
    return _llm_client


def _get_vision_client():
    """Vision client — try Groq vision model; fallback to OpenRouter."""
    global _vision_client
    if _vision_client is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if api_key:
            _vision_client = Groq(api_key=api_key)
    return _vision_client


# ---------------------------------------------------------------------------
# LLM calls with retry (Hard Rule 4 — tenacity for 429s)
# ---------------------------------------------------------------------------

def _call_llm_with_retry(client, model: str, messages: list, max_tokens: int = 512) -> str:
    """Call LLM with exponential backoff on 429. Returns content string."""
    from tenacity import (
        retry, stop_after_attempt, wait_exponential,
        retry_if_exception_type, before_sleep_log
    )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _inner():
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""

    return _inner()


# ---------------------------------------------------------------------------
# A. Image extraction (Hard Rule 2)
# ---------------------------------------------------------------------------

# System prompt for text extraction from OCR output
_IMAGE_EXTRACT_PROMPT = """You are a financial data extractor. Given OCR text from a financial document, 
extract ONLY the following fields. If a field is not present, return null.
Output ONLY valid JSON, no explanation.

{
  "amount": <number or null>,
  "currency": <3-letter ISO code or null>,
  "date": <"YYYY-MM-DD" or null>,
  "notes": <brief description or "">
}

Rules:
- amount must be a bare number (no currency symbols, no commas)
- currency must be the ISO code (INR, ZAR, IDR, USD, EUR)
- If you see amounts in multiple currencies, pick the one most likely to be the primary transaction amount
- Do NOT invent amounts. If genuinely unclear, return null for amount.
"""

# System prompt for vision extraction (fallback)
_VISION_EXTRACT_PROMPT = """You are a financial data extractor. Examine this image and extract ONLY:
{
  "amount": <number or null>,
  "currency": <3-letter ISO code or null>,
  "date": <"YYYY-MM-DD" or null>,
  "notes": <brief description or "">
}

Output ONLY valid JSON. Do not invent amounts — if unclear, return null.
"""


def extract_from_image(
    image_id: str,
    image_path: str,
    force_retry: bool = False,
) -> ImageExtraction:
    """
    Extract financial facts from an image.

    Hard Rule 2 pipeline:
      1. Run Tesseract OCR
      2. Pass OCR text to text LLM for structured extraction
      3. If OCR yields nothing usable, fall back to vision LLM
      4. Cache with status "ok" or "failed"
      5. NEVER treat blank amount as zero

    Hard Rule 4:
      - Cache entries with status "failed" are retried (force_retry=True or always re-attempt)
      - Only "ok" entries are returned from cache without re-running
    """
    cache = _load_cache(IMAGE_CACHE_FILE)

    # Check cache
    if image_id in cache and not force_retry:
        entry = cache[image_id]
        if entry.get("status") == "ok":
            logger.info("Image cache HIT (ok): %s", image_id)
            return ImageExtraction(
                image_id=image_id,
                amount=entry.get("amount"),
                currency=entry.get("currency"),
                date=entry.get("date"),
                notes=entry.get("notes", ""),
                source=entry.get("source", "cache"),
                raw_ocr=entry.get("raw_ocr", ""),
            )
        else:
            logger.info("Image cache HIT but status=%s — retrying %s", entry.get("status"), image_id)

    if not os.path.exists(image_path):
        logger.warning("Image file not found: %s", image_path)
        _save_cache(IMAGE_CACHE_FILE, {**cache, image_id: {"status": "failed", "reason": "file_not_found"}})
        return ImageExtraction(image_id=image_id, amount=None, currency=None, date=None,
                               notes="file not found", source="unresolvable")

    # Step 1: OCR
    ocr_text = run_ocr(image_path)
    logger.info("OCR for %s: %d chars", image_id, len(ocr_text))

    result: Optional[ImageExtraction] = None

    # Step 2: Text LLM from OCR
    if ocr_text and len(ocr_text) >= 5:
        try:
            client = _get_groq_client()
            model = os.environ.get("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
            user_msg = f"OCR text from financial document:\n\n{ocr_text}"
            content = _call_llm_with_retry(
                client, model,
                [
                    {"role": "system", "content": _IMAGE_EXTRACT_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=256,
            )
            parsed = _parse_json_response(content)
            if parsed and parsed.get("amount") is not None:
                result = ImageExtraction(
                    image_id=image_id,
                    amount=float(parsed["amount"]),
                    currency=parsed.get("currency"),
                    date=parsed.get("date"),
                    notes=parsed.get("notes", ""),
                    source="ocr_llm",
                    raw_ocr=ocr_text,
                )
                logger.info("OCR+LLM extraction succeeded for %s: amount=%.2f", image_id, result.amount)
        except Exception as e:
            logger.warning("OCR+LLM extraction failed for %s: %s", image_id, e)

    # Step 3: Vision LLM fallback
    if result is None or result.amount is None:
        logger.info("Falling back to vision LLM for %s", image_id)
        try:
            result = _extract_with_vision(image_id, image_path, ocr_text)
        except Exception as e:
            logger.warning("Vision LLM fallback failed for %s: %s", image_id, e)

    # Hard Rule 2 / Rule 3: NEVER treat blank as zero
    if result is None or result.amount is None:
        logger.warning(
            "HARD RULE 2/3: Could not resolve amount for image %s. "
            "Excluding event from forecast. NOT treating as zero.",
            image_id,
        )
        _save_cache(IMAGE_CACHE_FILE, {**cache, image_id: {
            "status": "failed",
            "reason": "unresolvable",
            "raw_ocr": ocr_text,
        }})
        return ImageExtraction(
            image_id=image_id, amount=None, currency=None, date=None,
            notes="unresolvable after OCR and vision", source="unresolvable",
        )

    # Save to cache as "ok"
    cache[image_id] = {
        "status": "ok",
        "amount": result.amount,
        "currency": result.currency,
        "date": result.date,
        "notes": result.notes,
        "source": result.source,
        "raw_ocr": result.raw_ocr,
    }
    _save_cache(IMAGE_CACHE_FILE, cache)
    return result


def _extract_with_vision(image_id: str, image_path: str, ocr_text: str) -> Optional[ImageExtraction]:
    """Vision LLM extraction using base64-encoded image."""
    import base64

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    client = _get_vision_client()
    if client is None:
        raise RuntimeError("No vision-capable LLM client available")

    model = os.environ.get("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

    content = _call_llm_with_retry(
        client, model,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_EXTRACT_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ],
            }
        ],
        max_tokens=256,
    )
    parsed = _parse_json_response(content)
    if parsed and parsed.get("amount") is not None:
        return ImageExtraction(
            image_id=image_id,
            amount=float(parsed["amount"]),
            currency=parsed.get("currency"),
            date=parsed.get("date"),
            notes=parsed.get("notes", ""),
            source="vision_llm",
        )
    return None


# ---------------------------------------------------------------------------
# B. Message extraction
# ---------------------------------------------------------------------------

_MSG_EXTRACT_PROMPT = """You are a financial amendment extractor. Given a message from an employer, bank, or service provider, 
extract any changes to the user's financial situation. Output ONLY valid JSON.

{
  "action": "<one of: none, salary_change, salary_date_change, rent_change, event_cancel, event_confirmed, refund_pending, refund_settled, investment_valuation, bonus_pending, income_ended, bonus_unconfirmed>",
  "new_amount": <number or null>,
  "new_date": <"YYYY-MM-DD" or null>,
  "currency": <"INR"|"ZAR"|"IDR"|"USD"|"EUR" or null>,
  "notes": <brief description or "">
}

Rules:
- "none" means the message contains no actionable financial amendment
- "salary_change" means a confirmed new recurring salary amount
- "salary_date_change" means the salary payment date has changed (not amount)
- "rent_change" means rent amount has changed
- "refund_pending" means a refund is initiated but NOT yet credited
- "refund_settled" means a refund HAS been credited to the account
- "income_ended" means a salary/income source has ended
- "bonus_pending" or "bonus_unconfirmed" means a bonus is NOT confirmed
- "investment_valuation" means market value change (non-cash, NOT available as cash)
- Per financial rules: pending credits, pending bonuses, unconfirmed commissions are NOT countable as income
- Do NOT invent information. If a message is ambiguous, return action="none"
"""


def extract_from_message(
    message_id: str,
    message_text: str,
    related_event_id: str = "",
    force_retry: bool = False,
) -> MessageAmendment:
    """
    Extract structured amendment from message text.
    Hard Rule 4: cache with explicit status.
    """
    cache = _load_cache(MSG_CACHE_FILE)

    if message_id in cache and not force_retry:
        entry = cache[message_id]
        if entry.get("status") == "ok":
            return MessageAmendment(
                message_id=message_id,
                action=entry.get("action", "none"),
                new_amount=entry.get("new_amount"),
                new_date=entry.get("new_date"),
                currency=entry.get("currency"),
                related_event_id=related_event_id,
                notes=entry.get("notes", ""),
            )
        else:
            logger.info("Message cache status=%s — retrying %s", entry.get("status"), message_id)

    try:
        client = _get_groq_client()
        model = os.environ.get("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
        content = _call_llm_with_retry(
            client, model,
            [
                {"role": "system", "content": _MSG_EXTRACT_PROMPT},
                {"role": "user", "content": f"Message:\n{message_text}"},
            ],
            max_tokens=256,
        )
        parsed = _parse_json_response(content)
        if parsed is None:
            raise ValueError("JSON parse failed")

        amend = MessageAmendment(
            message_id=message_id,
            action=parsed.get("action", "none"),
            new_amount=float(parsed["new_amount"]) if parsed.get("new_amount") is not None else None,
            new_date=parsed.get("new_date"),
            currency=parsed.get("currency"),
            related_event_id=related_event_id,
            notes=parsed.get("notes", ""),
        )

        cache[message_id] = {
            "status": "ok",
            "action": amend.action,
            "new_amount": amend.new_amount,
            "new_date": amend.new_date,
            "currency": amend.currency,
            "notes": amend.notes,
        }
        _save_cache(MSG_CACHE_FILE, cache)
        return amend

    except Exception as e:
        logger.warning("Message extraction failed for %s: %s", message_id, e)
        cache[message_id] = {"status": "failed", "reason": str(e)}
        _save_cache(MSG_CACHE_FILE, cache)
        return MessageAmendment(
            message_id=message_id,
            action="none",
            related_event_id=related_event_id,
            notes=f"extraction failed: {e}",
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> Optional[Dict]:
    """Extract and parse JSON from LLM response, tolerating markdown fences."""
    if not text:
        return None
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.strip("`").strip()
    # Find first { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
