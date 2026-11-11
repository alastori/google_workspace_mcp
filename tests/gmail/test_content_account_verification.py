"""A search account check cannot authorize a later service's body read."""

import base64
import inspect
from unittest.mock import AsyncMock, Mock

import pytest
import auth.service_decorator as auth
from core.server import server
from gmail.gmail_tools import (
    search_gmail_messages,
    get_gmail_message_content,
    get_gmail_messages_content_batch,
)

ACCOUNT = "fixture@example.invalid"
OTHER = "other@example.invalid"
TOOLS = [get_gmail_message_content, get_gmail_messages_content_batch]


class Batch:
    def __init__(self, callback):
        self.callback = callback
        self.requests = []

    def add(self, request, request_id):
        self.requests.append((request_id, request))

    def execute(self):
        for identifier, request in self.requests:
            self.callback(identifier, request.execute(), None)


def service(account):
    fake = Mock()
    fake.users.return_value.getProfile.return_value.execute.return_value = {
        "emailAddress": account
    }
    fake.users.return_value.messages.return_value.list.return_value.execute.return_value = {
        "messages": [{"id": "a1", "threadId": "b2"}]
    }
    fake.users.return_value.messages.return_value.get.return_value.execute.return_value = {
        "id": "a1",
        "payload": {
            "mimeType": "text/plain",
            "headers": [],
            "body": {"data": base64.urlsafe_b64encode(b"Fictional body").decode()},
        },
    }
    fake.new_batch_http_request.side_effect = lambda callback: Batch(callback)
    return fake


def wrapper(tool):
    function = tool.fn if hasattr(tool, "fn") else tool
    return function.__wrapped__


def patch_auth(monkeypatch, services, accounts, rewritten):
    monkeypatch.setattr(auth, "_user_email_is_managed", lambda: False)
    monkeypatch.setattr(
        auth,
        "_get_auth_context",
        AsyncMock(side_effect=[(account, "fictional", None) for account in accounts]),
    )
    monkeypatch.setattr(auth, "_detect_oauth_version", lambda *args: rewritten)
    authenticate = AsyncMock(side_effect=list(zip(services, accounts)))
    monkeypatch.setattr(auth, "_authenticate_service", authenticate)
    monkeypatch.setattr(auth, "_release_google_service_cycles", lambda: None)
    return authenticate


def arguments(tool, verified=True):
    result = dict(user_google_email=ACCOUNT, body_format="text")
    if tool is get_gmail_message_content:
        result.update(message_id="a1", full=False)
    else:
        result.update(message_ids=["a1"], format="full")
    # Run baseline behavior too, before the new optional arguments exist.
    if "require_verified_account" in inspect.signature(wrapper(tool)).parameters:
        result.update(require_verified_account=verified, expected_account=ACCOUNT)
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("rewritten", [False, True])
async def test_later_different_service_refuses_before_body_after_verified_search(
    monkeypatch, tool, rewritten
):
    first, second = service(ACCOUNT), service(OTHER)
    patch_auth(monkeypatch, [first, second], [ACCOUNT, OTHER], rewritten)
    result = await wrapper(search_gmail_messages)(
        query="Fictional",
        user_google_email=ACCOUNT,
        include_headers=False,
        require_verified_account=True,
        expected_account=ACCOUNT,
    )
    assert "Message ID: a1" in result

    with pytest.raises(ValueError, match="^gmail_authenticated_account_unavailable$"):
        await wrapper(tool)(**arguments(tool))

    second.users.return_value.getProfile.assert_called_once_with(
        userId="me", fields="emailAddress"
    )
    second.users.return_value.messages.return_value.get.assert_not_called()
    second.new_batch_http_request.assert_not_called()
    first.close.assert_called_once()
    second.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", TOOLS)
async def test_matching_content_service_checks_profile_and_preserves_text(
    monkeypatch, tool
):
    fake = service(ACCOUNT)
    patch_auth(monkeypatch, [fake], [ACCOUNT], False)
    result = await wrapper(tool)(**arguments(tool))
    assert "Fictional body" in result
    fake.users.return_value.getProfile.assert_called_once_with(
        userId="me", fields="emailAddress"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", TOOLS)
async def test_legacy_content_defaults_do_not_add_profile_request(monkeypatch, tool):
    fake = service(ACCOUNT)
    patch_auth(monkeypatch, [fake], [ACCOUNT], False)
    options = arguments(tool)
    options.pop("require_verified_account", None)
    options.pop("expected_account", None)
    result = await wrapper(tool)(**options)
    assert "Fictional body" in result
    fake.users.return_value.getProfile.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", ["get_gmail_message_content", "get_gmail_messages_content_batch"]
)
async def test_actual_content_schema_exposes_independent_verification(name):
    tools = {tool.name: tool for tool in await server.list_tools()}
    properties = tools[name].parameters["properties"]
    assert properties["require_verified_account"]["type"] == "boolean"
    assert properties["require_verified_account"]["default"] is False
    assert properties["expected_account"]["default"] is None
