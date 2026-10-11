"""Tests for explicit Stripe Checkout sessions and webhook handling."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi", reason="openjarvis[server] not installed")

from fastapi import FastAPI  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from openjarvis.server import billing_routes  # noqa: E402


class FakeStripeError(Exception):
    pass


class FakeSignatureVerificationError(Exception):
    pass


@pytest.fixture
def stripe_module(monkeypatch):
    session_create = MagicMock(
        return_value=SimpleNamespace(
            id="cs_test_123",
            url="https://checkout.stripe.com/c/pay/cs_test_123",
        )
    )
    client = SimpleNamespace(
        v1=SimpleNamespace(
            checkout=SimpleNamespace(sessions=SimpleNamespace(create=session_create))
        )
    )
    construct_event = MagicMock()
    module = SimpleNamespace(
        StripeClient=MagicMock(return_value=client),
        StripeError=FakeStripeError,
        Webhook=SimpleNamespace(construct_event=construct_event),
        error=SimpleNamespace(
            SignatureVerificationError=FakeSignatureVerificationError
        ),
    )
    monkeypatch.setitem(sys.modules, "stripe", module)
    return module, session_create, construct_event


@pytest.fixture
def client(monkeypatch, stripe_module, tmp_path):
    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_example")
    monkeypatch.setenv("OPENJARVIS_STRIPE_PAYMENT_PRICE_ID", "price_test_one_time")
    monkeypatch.setenv(
        "OPENJARVIS_STRIPE_SUBSCRIPTION_PRICE_ID", "price_test_recurring"
    )
    monkeypatch.setenv(
        "OPENJARVIS_STRIPE_SUCCESS_URL",
        "https://example.test/billing/success?session_id={CHECKOUT_SESSION_ID}",
    )
    monkeypatch.setenv(
        "OPENJARVIS_STRIPE_CANCEL_URL", "https://example.test/billing/cancel"
    )
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    app = FastAPI()
    app.include_router(billing_routes.billing_router)
    app.include_router(billing_routes.stripe_webhook_router)
    return TestClient(app)


@pytest.mark.parametrize(
    ("mode", "price_id"),
    [
        ("payment", "price_test_one_time"),
        ("subscription", "price_test_recurring"),
    ],
)
def test_checkout_session_uses_server_configured_price(
    client, stripe_module, mode, price_id
):
    _, session_create, _ = stripe_module

    response = client.post(
        "/v1/billing/checkout-session",
        json={"mode": mode},
    )

    assert response.status_code == 200
    assert response.json() == {
        "checkout_url": "https://checkout.stripe.com/c/pay/cs_test_123",
        "session_id": "cs_test_123",
        "mode": mode,
    }
    session_create.assert_called_once_with(
        {
            "mode": mode,
            "line_items": [{"price": price_id, "quantity": 1}],
            "success_url": (
                "https://example.test/billing/success?session_id={CHECKOUT_SESSION_ID}"
            ),
            "cancel_url": "https://example.test/billing/cancel",
        }
    )


def test_checkout_rejects_client_supplied_price_id(client, stripe_module):
    _, session_create, _ = stripe_module

    response = client.post(
        "/v1/billing/checkout-session",
        json={"mode": "payment", "price_id": "price_attacker_selected"},
    )

    assert response.status_code == 422
    session_create.assert_not_called()


def test_checkout_disables_live_mode(client, monkeypatch, stripe_module):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_live_not-a-real-key")
    _, session_create, _ = stripe_module

    response = client.post(
        "/v1/billing/checkout-session",
        json={"mode": "payment"},
    )

    assert response.status_code == 503
    assert "live mode is disabled" in response.json()["detail"]
    session_create.assert_not_called()


def test_checkout_requires_configured_price_id(client, monkeypatch, stripe_module):
    monkeypatch.delenv("OPENJARVIS_STRIPE_PAYMENT_PRICE_ID")
    _, session_create, _ = stripe_module

    response = client.post(
        "/v1/billing/checkout-session",
        json={"mode": "payment"},
    )

    assert response.status_code == 503
    session_create.assert_not_called()


def test_checkout_does_not_expose_stripe_provider_error(
    client, stripe_module, monkeypatch
):
    _, session_create, _ = stripe_module
    session_create.side_effect = FakeStripeError("sensitive provider detail")

    response = client.post(
        "/v1/billing/checkout-session",
        json={"mode": "payment"},
    )

    assert response.status_code == 502
    assert "sensitive provider detail" not in response.text


def test_webhook_verifies_snapshot_event_and_deduplicates(
    client, monkeypatch, stripe_module, tmp_path
):
    _, _, construct_event = stripe_module
    event = {
        "id": "evt_test_123",
        "type": "checkout.session.completed",
        "livemode": False,
        "data": {"object": {"id": "cs_test_123"}},
    }
    construct_event.return_value = event
    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))

    first = client.post(
        "/webhooks/stripe",
        content=b'{"id":"evt_test_123"}',
        headers={"stripe-signature": "valid-signature"},
    )
    duplicate = client.post(
        "/webhooks/stripe",
        content=b'{"id":"evt_test_123"}',
        headers={"stripe-signature": "valid-signature"},
    )

    assert first.status_code == 200
    assert first.json() == {"received": True, "duplicate": False}
    assert duplicate.status_code == 200
    assert duplicate.json() == {"received": True, "duplicate": True}
    construct_event.assert_called_with(
        b'{"id":"evt_test_123"}',
        "valid-signature",
        "whsec_test",
    )


def test_webhook_rejects_invalid_signature(client, stripe_module):
    _, _, construct_event = stripe_module
    construct_event.side_effect = FakeSignatureVerificationError("invalid")

    response = client.post(
        "/webhooks/stripe",
        content=b"{}",
        headers={"stripe-signature": "invalid"},
    )

    assert response.status_code == 400


def test_webhook_rejects_live_mode_event(client, stripe_module, tmp_path, monkeypatch):
    _, _, construct_event = stripe_module
    construct_event.return_value = {
        "id": "evt_live_123",
        "type": "checkout.session.completed",
        "livemode": True,
        "data": {"object": {"id": "cs_live_123"}},
    }
    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))

    response = client.post(
        "/webhooks/stripe",
        content=b"{}",
        headers={"stripe-signature": "valid-signature"},
    )

    assert response.status_code == 400
