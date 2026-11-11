"""Empty Gmail pages must retain provider continuation tokens."""

import inspect
from unittest.mock import Mock

import pytest

from gmail.gmail_tools import _format_gmail_results_plain, search_gmail_messages


def test_empty_page_with_continuation_keeps_existing_pagination_footer():
    result = _format_gmail_results_plain([], "Fictional", "next2")
    assert result.startswith("Found 0 messages matching 'Fictional':\n\n📧 MESSAGES:\n")
    assert result.endswith("page_token='next2'")
    assert "💡 USAGE:" in result


def test_empty_page_without_continuation_preserves_existing_output():
    assert _format_gmail_results_plain([], "Fictional") == (
        "No messages found for query: 'Fictional'"
    )


@pytest.mark.parametrize(
    "token", ["", False, 17, "bad' token", "bad\nnext", "x" * 2049]
)
def test_malformed_pagination_refuses_instead_of_appearing_exhausted(token):
    with pytest.raises(ValueError, match="invalid Gmail pagination token"):
        _format_gmail_results_plain([], "Fictional", token)


@pytest.mark.asyncio
async def test_actual_search_keeps_empty_provider_page_token_without_metadata():
    service = Mock()
    service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
        "messages": [],
        "nextPageToken": "next2",
    }
    function = (
        search_gmail_messages.fn
        if hasattr(search_gmail_messages, "fn")
        else search_gmail_messages
    )
    result = await inspect.unwrap(function)(
        service=service,
        query="Fictional",
        user_google_email="fixture@example.invalid",
        include_headers=False,
    )
    assert "Found 0 messages" in result
    assert result.endswith("page_token='next2'")
    service.users.return_value.messages.return_value.get.assert_not_called()
    service.new_batch_http_request.assert_not_called()
