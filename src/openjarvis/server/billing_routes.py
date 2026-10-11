"""Stripe Checkout and webhook routes.

Checkout is configured entirely by server-side Stripe Price IDs. This module
does not expose card data, customer identity, or arbitrary prices to clients.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from openjarvis.core.paths import get_data_dir

logger = logging.getLogger(__name__)

billing_router = APIRouter(prefix="/v1/billing", tags=["billing"])
stripe_webhook_router = APIRouter(prefix="/webhooks", tags=["webhooks"])


class CheckoutSessionRequest(BaseModel):
    """The allowed Stripe Checkout operation; price selection stays server-side."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["payment", "subscription"]


class StripeUnavailable(RuntimeError):
    """The Stripe SDK is unavailable in this server installation."""


class StripeProviderFailure(RuntimeError):
    """A Stripe API call failed without exposing provider details to the caller."""


class InvalidStripeSignature(ValueError):
    """The Stripe webhook signature or event payload is invalid."""


def _stripe_sdk() -> Any:
    try:
        import stripe
    except ImportError as exc:
        raise StripeUnavailable from exc
    return stripe


def _stripe_secret() -> str:
    from openjarvis.core.credentials import get_tool_credential

    return (get_tool_credential("stripe", "STRIPE_SECRET_KEY") or "").strip()


def _stripe_webhook_secret() -> str:
    from openjarvis.core.credentials import get_tool_credential

    return (get_tool_credential("stripe", "STRIPE_WEBHOOK_SECRET") or "").strip()


def _configured_return_url(name: str) -> str:
    value = os.environ.get(name, "").strip()
    try:
        parsed = urlsplit(value)
    except ValueError:
        parsed = None
    if (
        not value
        or parsed is None
        or parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise HTTPException(
            status_code=503,
            detail=f"Set a valid absolute URL in {name} before using Stripe Checkout",
        )
    return value


def _call_stripe_checkout(secret_key: str, params: dict[str, Any]) -> Any:
    stripe = _stripe_sdk()
    try:
        client = stripe.StripeClient(secret_key)
        return client.v1.checkout.sessions.create(params)
    except stripe.StripeError as exc:
        raise StripeProviderFailure from exc


def _construct_webhook_event(payload: bytes, signature: str, secret: str) -> Any:
    stripe = _stripe_sdk()
    try:
        return stripe.Webhook.construct_event(payload, signature, secret)
    except ValueError as exc:
        raise InvalidStripeSignature from exc
    except stripe.error.SignatureVerificationError as exc:
        raise InvalidStripeSignature from exc


def _event_field(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


class StripeEventStore:
    """Persist a minimal, durable event-id ledger for Stripe retry deduplication."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or (get_data_dir() / "stripe_events.sqlite3")

    def record(self, *, event_id: str, event_type: str, object_id: str | None) -> bool:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path), timeout=5) as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS stripe_webhook_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    object_id TEXT,
                    received_at TEXT NOT NULL
                )
                """
            )
            cursor = db.execute(
                """
                INSERT OR IGNORE INTO stripe_webhook_events
                    (event_id, event_type, object_id, received_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    event_id,
                    event_type,
                    object_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return cursor.rowcount == 1


@billing_router.post("/checkout-session")
def create_checkout_session(body: CheckoutSessionRequest):
    """Create a hosted Checkout Session for a configured payment or subscription."""
    secret_key = _stripe_secret()
    if not secret_key:
        raise HTTPException(
            status_code=503,
            detail="Stripe is not configured: set STRIPE_SECRET_KEY",
        )
    if not secret_key.startswith(("sk_test_", "rk_test_")):
        raise HTTPException(
            status_code=503,
            detail="Stripe live mode is disabled; configure a test-mode secret key",
        )

    price_env = (
        "OPENJARVIS_STRIPE_PAYMENT_PRICE_ID"
        if body.mode == "payment"
        else "OPENJARVIS_STRIPE_SUBSCRIPTION_PRICE_ID"
    )
    price_id = os.environ.get(price_env, "").strip()
    if not price_id.startswith("price_"):
        raise HTTPException(
            status_code=503,
            detail=f"Set a Stripe test-mode Price ID in {price_env}",
        )

    success_url = _configured_return_url("OPENJARVIS_STRIPE_SUCCESS_URL")
    if "{CHECKOUT_SESSION_ID}" not in success_url:
        raise HTTPException(
            status_code=503,
            detail=("OPENJARVIS_STRIPE_SUCCESS_URL must include {CHECKOUT_SESSION_ID}"),
        )
    cancel_url = _configured_return_url("OPENJARVIS_STRIPE_CANCEL_URL")
    params = {
        "mode": body.mode,
        "line_items": [{"price": price_id, "quantity": 1}],
        "success_url": success_url,
        "cancel_url": cancel_url,
    }

    try:
        session = _call_stripe_checkout(secret_key, params)
    except StripeUnavailable:
        raise HTTPException(
            status_code=503,
            detail="Stripe support is unavailable; install the server dependencies",
        ) from None
    except StripeProviderFailure:
        logger.warning("Stripe Checkout session creation failed")
        raise HTTPException(
            status_code=502,
            detail="Stripe could not create a Checkout Session",
        ) from None

    checkout_url = _event_field(session, "url")
    session_id = _event_field(session, "id")
    if (
        not isinstance(checkout_url, str)
        or not checkout_url.startswith("https://checkout.stripe.com/")
        or not isinstance(session_id, str)
        or not session_id.startswith("cs_")
    ):
        logger.warning("Stripe returned a Checkout Session without a hosted URL")
        raise HTTPException(
            status_code=502,
            detail="Stripe returned an invalid Checkout Session",
        )
    return {
        "checkout_url": checkout_url,
        "session_id": session_id,
        "mode": body.mode,
    }


@stripe_webhook_router.post("/stripe")
async def receive_stripe_webhook(
    request: Request,
    stripe_signature: str = Header(alias="stripe-signature"),
):
    """Verify and durably deduplicate a Stripe snapshot webhook event."""
    secret = _stripe_webhook_secret()
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="Stripe webhooks are not configured: set STRIPE_WEBHOOK_SECRET",
        )
    payload = await request.body()
    try:
        event = _construct_webhook_event(payload, stripe_signature, secret)
    except StripeUnavailable:
        raise HTTPException(
            status_code=503,
            detail="Stripe support is unavailable; install the server dependencies",
        ) from None
    except InvalidStripeSignature:
        raise HTTPException(status_code=400, detail="Invalid Stripe webhook signature")

    event_id = _event_field(event, "id")
    event_type = _event_field(event, "type")
    if (
        not isinstance(event_id, str)
        or not event_id
        or not isinstance(event_type, str)
        or not event_type
        or _event_field(event, "livemode") is not False
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid or live-mode Stripe webhook event",
        )

    data = _event_field(event, "data")
    stripe_object = _event_field(data, "object")
    object_id = _event_field(stripe_object, "id")
    if not isinstance(object_id, str):
        object_id = None

    inserted = StripeEventStore().record(
        event_id=event_id,
        event_type=event_type,
        object_id=object_id,
    )
    return {"received": True, "duplicate": not inserted}


__all__ = ["billing_router", "stripe_webhook_router"]
