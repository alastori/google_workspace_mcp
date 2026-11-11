"""Account verification and location contracts through actual Gmail modules."""

import inspect
from unittest.mock import AsyncMock, Mock

import pytest

import auth.service_decorator as service_decorator
from core.server import server
from gmail.gmail_helpers import _require_verified_gmail_account
from gmail.gmail_tools import search_gmail_messages

ACCOUNT = "fixture@example.invalid"
OTHER_ACCOUNT = "substituted@example.invalid"
QUERY = "after:1793923200 before:1794009600 subject:Fictional"


def _unwrap(tool):
    function = tool.fn if hasattr(tool, "fn") else tool
    return inspect.unwrap(function)


@pytest.fixture
def service():
    fake = Mock()
    fake.users.return_value.getProfile.return_value.execute.return_value = {
        "emailAddress": ACCOUNT
    }
    fake.users.return_value.messages.return_value.list.return_value.execute.return_value = {
        "messages": [{"id": "a1", "threadId": "b2"}]
    }
    return fake


async def _search(service, **overrides):
    arguments = dict(
        service=service,
        query=QUERY,
        user_google_email=ACCOUNT,
        require_verified_account=True,
        expected_account=ACCOUNT,
    )
    arguments.update(overrides)
    return await _unwrap(search_gmail_messages)(**arguments)


@pytest.mark.asyncio
async def test_helper_requests_only_authenticated_email(service):
    await _require_verified_gmail_account(service, ACCOUNT)

    service.users.return_value.getProfile.assert_called_once_with(
        userId="me", fields="emailAddress"
    )
    service.users.return_value.messages.return_value.list.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("expected", [None, "", 17, "bad address", ACCOUNT + "\n"])
async def test_invalid_expected_account_refuses_before_profile(service, expected):
    with pytest.raises(ValueError, match="^gmail_authenticated_account_unavailable$"):
        await _search(service, expected_account=expected)

    service.users.return_value.getProfile.assert_not_called()
    service.users.return_value.messages.return_value.list.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profile",
    [
        None,
        {},
        [],
        {"emailAddress": None},
        {"emailAddress": 17},
        {"emailAddress": OTHER_ACCOUNT},
        {"emailAddress": ACCOUNT + "\n"},
        {"emailAddress": "fixture+alias@example.invalid"},
    ],
)
async def test_unverified_profile_refuses_before_search(service, profile):
    service.users.return_value.getProfile.return_value.execute.return_value = profile

    with pytest.raises(ValueError, match="^gmail_authenticated_account_unavailable$"):
        await _search(service)

    service.users.return_value.messages.return_value.list.assert_not_called()


@pytest.mark.asyncio
async def test_profile_failure_has_fixed_diagnostic(service):
    service.users.return_value.getProfile.return_value.execute.side_effect = (
        RuntimeError("PRIVATE fictional provider diagnostic")
    )

    with pytest.raises(
        ValueError, match="^gmail_authenticated_account_unavailable$"
    ) as error:
        await _search(service)

    assert "PRIVATE" not in str(error.value)
    assert error.value.__suppress_context__ is True
    service.users.return_value.messages.return_value.list.assert_not_called()


@pytest.mark.asyncio
async def test_case_only_account_difference_is_accepted(service):
    service.users.return_value.getProfile.return_value.execute.return_value = {
        "emailAddress": "Fixture@Example.Invalid"
    }

    await _search(service)

    service.users.return_value.messages.return_value.list.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("include_spam_trash", [False, True])
async def test_verified_headerless_search_preserves_location_and_order(
    service, include_spam_trash
):
    events = []
    profile = {"emailAddress": ACCOUNT}
    messages = {"messages": [{"id": "a1", "threadId": "b2"}]}
    service.users.return_value.getProfile.return_value.execute.side_effect = lambda: (
        events.append("profile") or profile
    )
    service.users.return_value.messages.return_value.list.return_value.execute.side_effect = (
        lambda: events.append("search") or messages
    )

    result = await _search(
        service, include_headers=False, include_spam_trash=include_spam_trash
    )

    assert events == ["profile", "search"]
    service.users.return_value.messages.return_value.list.assert_called_once_with(
        userId="me", q=QUERY, maxResults=10, includeSpamTrash=include_spam_trash
    )
    service.users.return_value.messages.return_value.get.assert_not_called()
    service.new_batch_http_request.assert_not_called()
    assert "Message ID: a1" in result
    assert "Thread ID: b2" in result
    assert "Subject:" not in result
    assert "From:" not in result
    assert "Date:" not in result


@pytest.mark.asyncio
async def test_each_search_rechecks_authenticated_profile(service):
    await _search(service)
    service.users.return_value.getProfile.return_value.execute.return_value = {
        "emailAddress": OTHER_ACCOUNT
    }

    with pytest.raises(ValueError, match="^gmail_authenticated_account_unavailable$"):
        await _search(service)

    assert service.users.return_value.getProfile.call_count == 2
    service.users.return_value.messages.return_value.list.assert_called_once()


@pytest.mark.asyncio
async def test_existing_call_defaults_do_not_add_profile_request(service):
    await _unwrap(search_gmail_messages)(
        service=service, query=QUERY, user_google_email=ACCOUNT
    )

    service.users.return_value.getProfile.assert_not_called()
    service.users.return_value.messages.return_value.list.assert_called_once_with(
        userId="me", q=QUERY, maxResults=10, includeSpamTrash=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [None, 0, 1, "true", "false"])
async def test_non_boolean_verification_refuses_before_service_calls(service, flag):
    with pytest.raises(ValueError, match="require_verified_account must be a Boolean"):
        await _search(service, require_verified_account=flag)

    service.users.return_value.getProfile.assert_not_called()
    service.users.return_value.messages.return_value.list.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [None, 0, 1, "true", "false"])
async def test_non_boolean_location_refuses_before_search(service, flag):
    with pytest.raises(ValueError, match="include_spam_trash must be a Boolean"):
        await _search(service, require_verified_account=False, include_spam_trash=flag)

    service.users.return_value.getProfile.assert_not_called()
    service.users.return_value.messages.return_value.list.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("authenticated_account", [ACCOUNT, OTHER_ACCOUNT])
async def test_auth_wrapper_preserves_independent_expected_account(
    monkeypatch, service, managed, authenticated_account
):
    monkeypatch.setattr(service_decorator, "_user_email_is_managed", lambda: managed)
    monkeypatch.setattr(
        service_decorator,
        "_get_auth_context",
        AsyncMock(return_value=(authenticated_account, "fictional", None)),
    )
    monkeypatch.setattr(service_decorator, "_detect_oauth_version", lambda *args: True)
    monkeypatch.setattr(
        service_decorator,
        "_extract_managed_user_email",
        lambda *args: authenticated_account,
    )
    authenticate = AsyncMock(return_value=(service, authenticated_account))
    monkeypatch.setattr(service_decorator, "_authenticate_service", authenticate)
    monkeypatch.setattr(
        service_decorator, "_release_google_service_cycles", lambda: None
    )
    service.users.return_value.getProfile.return_value.execute.return_value = {
        "emailAddress": authenticated_account
    }
    # Keep the real auth wrapper and its account substitution. Skip only the outer
    # HTTP error formatting decorator, so the source refusal remains observable.
    tool_function = (
        search_gmail_messages.fn
        if hasattr(search_gmail_messages, "fn")
        else search_gmail_messages
    )
    authenticated_search = tool_function.__wrapped__
    arguments = dict(
        query=QUERY,
        user_google_email=ACCOUNT,
        require_verified_account=True,
        expected_account=ACCOUNT,
    )

    if authenticated_account == ACCOUNT:
        result = await authenticated_search(**arguments)
        assert "Message ID: a1" in result
        service.users.return_value.messages.return_value.list.assert_called_once()
    else:
        with pytest.raises(
            ValueError, match="^gmail_authenticated_account_unavailable$"
        ):
            await authenticated_search(**arguments)
        service.users.return_value.messages.return_value.list.assert_not_called()

    authenticate.assert_awaited_once()
    assert authenticate.call_args.args[4] == authenticated_account
    service.users.return_value.getProfile.assert_called_once_with(
        userId="me", fields="emailAddress"
    )
    service.close.assert_called_once()


@pytest.mark.asyncio
async def test_published_search_schema_exposes_scoped_options():
    tools = {tool.name: tool for tool in await server.list_tools()}
    properties = tools["search_gmail_messages"].parameters["properties"]

    for name in ("require_verified_account", "include_spam_trash", "include_headers"):
        assert properties[name]["type"] == "boolean"
        assert properties[name]["default"] is False
    assert properties["expected_account"]["default"] is None
    assert {item["type"] for item in properties["expected_account"]["anyOf"]} == {
        "string",
        "null",
    }
