"""Issued matching-message review uses no thread-member discovery."""

import base64
import json
from types import SimpleNamespace as NS
from urllib.parse import parse_qs, urlencode
import httpx
import pytest
from core import file_limits
from gmail import gmail_helpers as helper
from gmail.gmail_tools import get_gmail_messages_content_batch

ACCOUNT = "fixture@example.invalid"
START = 1793923200000
END = 1794009600000
QUERY = 'after:1793923200 before:1794009600 from:person@example.invalid "Fictional" -(filename:"ics") in:anywhere'
SCOPE = dict(
    since="2026-11-06T00:00:00Z",
    until="2026-11-07T00:00:00Z",
    members={"a1": "b1"},
    query_provenance={"a1": [QUERY]},
    issued_thread_ids=["b1", "b2"],
    observed_message_count=2,
)


class Credentials:
    token = "fictional-test-token"
    valid = True
    expired = False

    def apply(self, headers):
        headers["authorization"] = "Bearer " + self.token

    def refresh(self, *args):
        raise AssertionError("refresh forbidden")


class Service:
    def __init__(self):
        self.credentials = Credentials()

    def users(self):
        return self

    def messages(self):
        return self

    def threads(self):
        raise AssertionError("unseen thread discovery forbidden")

    def list(self, **kwargs):
        raise AssertionError("new search forbidden")

    def getProfile(self, **kwargs):
        assert kwargs == dict(userId="me", fields="emailAddress")
        return self.request("profile", kwargs)

    def get(self, **kwargs):
        return self.request(
            "date" if kwargs["fields"] == "id,threadId,internalDate" else "body", kwargs
        )

    def request(self, kind, kwargs):
        return NS(
            method="GET",
            uri="https://gmail.googleapis.com/gmail/v1/users/me/"
            + kind
            + "?"
            + urlencode(kwargs),
            headers={},
            http=NS(credentials=self.credentials),
        )

    def close(self):
        pass


def _api_state(monkeypatch):
    state = NS(
        profile=ACCOUNT,
        date=dict(id="a1", threadId="b1", internalDate=str(START)),
        body=dict(
            id="a1",
            threadId="b1",
            internalDate=str(START),
            payload=dict(
                mimeType="text/plain",
                body=dict(
                    data=base64.urlsafe_b64encode(
                        b"Fictional matching message"
                    ).decode()
                ),
            ),
        ),
        calls=[],
    )
    state.by_id = None
    real = httpx.AsyncClient

    def respond(request):
        state.calls.append(request)
        kind = request.url.path.rsplit("/", 1)[-1]
        params = parse_qs(request.url.query.decode())
        if kind == "profile":
            assert params["fields"] == ["emailAddress"]
            value = dict(emailAddress=state.profile)
        elif kind == "date":
            assert params["fields"] == ["id,threadId,internalDate"]
            mid = params["id"][0]
            value = state.by_id[mid] if state.by_id is not None else state.date
        else:
            assert params["fields"] == ["id,threadId,internalDate,payload"]
            mid = params["id"][0]
            value = (
                dict(state.by_id[mid], payload=state.body["payload"])
                if state.by_id is not None
                else state.body
            )

        class Entity(httpx.AsyncByteStream):
            async def __aiter__(self):
                raw = value if isinstance(value, bytes) else json.dumps(value).encode()
                for offset in range(0, len(raw), 65536):
                    yield raw[offset : offset + 65536]

        return httpx.Response(200, stream=Entity())

    transport = httpx.MockTransport(respond)

    def client(**kwargs):
        kwargs["transport"] = transport
        return real(**kwargs)

    monkeypatch.setattr(file_limits.httpx, "AsyncClient", client)
    return state


@pytest.fixture
def api(monkeypatch):
    return _api_state(monkeypatch)


@pytest.mark.asyncio
async def test_matching_only_groups_and_omissions(api):
    result = await helper._review_matching_gmail(Service(), ["a1"], SCOPE, ACCOUNT)
    assert result["coverage"] == "matching_messages_only"
    assert result["groups"] == [dict(thread_id="b1", message_ids=["a1"])]
    assert result["omitted_thread_ids"] == ["b2"]
    assert result["items"][0]["text"] == "Fictional matching message"
    assert result["unavailable_messages"] == []
    assert len(api.calls) == 4
    assert "unseen_replies_not_enumerated" in result["limitations"]


@pytest.mark.asyncio
@pytest.mark.parametrize("timestamp", [START - 1, END])
async def test_date_gate_precedes_every_body(api, timestamp):
    api.date["internalDate"] = str(timestamp)
    result = await helper._review_matching_gmail(Service(), ["a1"], SCOPE, ACCOUNT)
    assert not result["items"]
    assert result["unavailable_messages"] == [dict(id="a1", reason="outside_window")]
    assert len(api.calls) == 2
    assert all("/body?" not in str(call.url) for call in api.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value", [("id", "a2"), ("threadId", "b2"), ("internalDate", str(END))]
)
async def test_body_identity_date_drift_refuses(api, field, value):
    api.body[field] = value
    with pytest.raises(ValueError):
        await helper._review_matching_gmail(Service(), ["a1"], SCOPE, ACCOUNT)


@pytest.mark.asyncio
async def test_unissued_argument_or_thread_expansion_refuses(api):
    with pytest.raises(ValueError):
        await helper._review_matching_gmail(Service(), ["a2"], SCOPE, ACCOUNT)
    with pytest.raises(ValueError):
        await helper._review_matching_gmail(
            Service(), ["a1"], dict(SCOPE, additional_thread_ids=["b2"]), ACCOUNT
        )
    assert not api.calls


@pytest.mark.asyncio
async def test_missing_date_and_wrong_account_refuse(api):
    api.date.pop("internalDate")
    with pytest.raises(ValueError):
        await helper._review_matching_gmail(Service(), ["a1"], SCOPE, ACCOUNT)
    api.profile = "other@example.invalid"
    with pytest.raises(ValueError):
        await helper._review_matching_gmail(Service(), ["a1"], SCOPE, ACCOUNT)
    assert all("/body?" not in str(call.url) for call in api.calls)


@pytest.mark.asyncio
async def test_public_batch_wrapper_no_refresh_and_provider_packet(api, monkeypatch):
    from unittest.mock import AsyncMock
    import auth.service_decorator as auth

    service = Service()
    authenticate = AsyncMock(return_value=(service, ACCOUNT))
    monkeypatch.setattr(
        auth, "_get_auth_context", AsyncMock(return_value=(ACCOUNT, "fictional", None))
    )
    monkeypatch.setattr(auth, "_detect_oauth_version", lambda *args: False)
    monkeypatch.setattr(auth, "_user_email_is_managed", lambda: False)
    monkeypatch.setattr(auth, "_authenticate_service", authenticate)
    function = (
        get_gmail_messages_content_batch.fn
        if hasattr(get_gmail_messages_content_batch, "fn")
        else get_gmail_messages_content_batch
    )
    packet = await function(
        message_ids=["a1"],
        user_google_email=ACCOUNT,
        format="full",
        body_format="text",
        require_verified_account=True,
        expected_account=ACCOUNT,
        matching_review=SCOPE,
    )
    assert authenticate.call_args.kwargs["allow_refresh"] is False
    result = json.loads(packet)
    assert result["items"][0]["id"] == "a1"
    assert result["items"][0]["thread_id"] == "b1"
    assert result["body_reads"] == result["metadata_reads"] == 1
    assert result["gmail_requests"] == len(api.calls) == 4
    assert result["omitted_thread_ids"] == ["b2"]


@pytest.mark.asyncio
async def test_unavailable_text_is_explicit_and_not_empty_mailbox(api):
    api.body["payload"] = dict(
        mimeType="text/html",
        body=dict(data=base64.urlsafe_b64encode(b"<p>Fictional</p>").decode()),
    )
    result = await helper._review_matching_gmail(Service(), ["a1"], SCOPE, ACCOUNT)
    assert not result["items"]
    assert result["body_reads"] == 1
    assert result["unavailable_messages"] == [
        dict(id="a1", reason="text_plain_unavailable")
    ]
    assert "reply_completeness_unavailable" in result["limitations"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entity",
    [
        b'{"id":"a1","id":"a1"}',
        b"\xff",
        b'{"id":NaN}',
        json.dumps(dict(id="a1")).encode("utf-16"),
    ],
)
async def test_metadata_json_is_strict_before_body(api, entity):
    api.date = entity
    with pytest.raises(ValueError):
        await helper._review_matching_gmail(Service(), ["a1"], SCOPE, ACCOUNT)
    assert all("/body?" not in str(call.url) for call in api.calls)


@pytest.mark.asyncio
async def test_expired_credentials_never_refresh(api):
    service = Service()
    service.credentials.expired = True
    with pytest.raises(ValueError):
        await helper._review_matching_gmail(service, ["a1"], SCOPE, ACCOUNT)
    assert not api.calls


@pytest.mark.asyncio
async def test_twenty_selected_ids_stop_at_request_cap_with_truthful_omissions(api):
    ids = [format(index + 1, "x") for index in range(20)]
    members = {mid: ["b1", "b2", "b3"][index % 3] for index, mid in enumerate(ids)}
    api.by_id = {
        mid: dict(id=mid, threadId=tid, internalDate=str(START))
        for mid, tid in members.items()
    }
    scope = dict(
        SCOPE,
        members=members,
        query_provenance={mid: [QUERY] for mid in ids},
        issued_thread_ids=["b1", "b2", "b3", "b4"],
        observed_message_count=21,
    )
    result = await helper._review_matching_gmail(Service(), ids, scope, ACCOUNT)
    assert result["body_reads"] == 11
    assert result["gmail_requests"] == len(api.calls) == 44
    assert len(result["unavailable_messages"]) == 9
    assert {row["reason"] for row in result["unavailable_messages"]} == {
        "budget_exhausted"
    }
    assert result["selected_thread_ids"] == ["b1", "b2", "b3"]
    assert result["omitted_thread_ids"] == ["b4"]
    assert result["matching_messages_not_selected"] == 1
    assert result["response_entity_bytes"] <= 8 * 1024**2
    assert all(
        params[0] in ids
        for request in api.calls
        if (params := parse_qs(request.url.query.decode()).get("id"))
    )


@pytest.mark.asyncio
async def test_unsupported_mime_is_explicit_content_gap(api):
    api.body["payload"]["body"]["data"] = "!"
    result = await helper._review_matching_gmail(Service(), ["a1"], SCOPE, ACCOUNT)
    assert result["items"] == []
    assert result["unavailable_messages"] == [
        dict(id="a1", reason="content_unavailable")
    ]
    assert result["body_reads"] == 1


async def _partial_packet(api, mode):
    ids = [format(index + 1, "x") for index in range(20 if mode == "budget" else 3)]
    members = dict.fromkeys(ids, "b1")
    api.by_id = {
        mid: dict(
            id=mid,
            threadId="b1",
            internalDate=str(END if mode == "outside" else START),
        )
        for mid in ids
    }
    if mode == "html":
        api.body["payload"]["mimeType"] = "text/html"
    elif mode == "malformed":
        api.body["payload"]["body"]["data"] = "!"
    elif mode == "output":
        api.body["payload"]["body"]["data"] = base64.urlsafe_b64encode(
            b"x" * 100000
        ).decode()
    scope = dict(
        SCOPE,
        members=members,
        query_provenance={mid: [QUERY] for mid in ids},
        issued_thread_ids=["b1"],
        observed_message_count=len(ids),
    )
    result = await helper._review_matching_gmail(Service(), ids, scope, ACCOUNT)
    return result, ids, members


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,reason,metadata,bodies",
    [
        ("html", "text_plain_unavailable", 3, 3),
        ("malformed", "content_unavailable", 3, 3),
        ("outside", "outside_window", 3, 0),
        ("output", "output_limit", 2, 2),
        ("budget", "budget_exhausted", 11, 11),
    ],
)
async def test_actual_partial_packets_have_source_accounting(
    api, mode, reason, metadata, bodies
):
    result, _, _ = await _partial_packet(api, mode)
    assert result["metadata_reads"] == metadata
    assert result["body_reads"] == bodies
    assert result["gmail_requests"] == len(api.calls) == 2 * (metadata + bodies)
    assert {row["reason"] for row in result["unavailable_messages"]} == {reason}
