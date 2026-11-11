"""Fictional transport assertions for the opt-in shared acquisition budget."""

import asyncio
import unittest
from unittest.mock import patch

import httpx
from core import file_limits as target

URL = "https://fictional.example.invalid/data"
READ = 65536


class Stream(httpx.AsyncByteStream):
    def __init__(self, chunks, delay=0):
        self.chunks = chunks
        self.delay = delay
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk

    async def aclose(self):
        self.closed = True


class StrictBudgetTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._prior_policy = asyncio.get_event_loop_policy()
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    @classmethod
    def tearDownClass(cls):
        try:
            asyncio.set_event_loop_policy(cls._prior_policy)
        finally:
            del cls._prior_policy
            super().tearDownClass()

    async def request(self, budget, stream, *, status=200, headers=None, limit=16):
        calls = []
        real_client = httpx.AsyncClient

        def respond(request):
            calls.append(request)
            self.assertEqual(request.headers["accept-encoding"], "identity")
            return httpx.Response(status, headers=headers, stream=stream)

        transport = httpx.MockTransport(respond)

        def client(**kwargs):
            self.assertFalse(kwargs["follow_redirects"])
            self.assertFalse(kwargs["trust_env"])
            self.assertFalse(kwargs["http2"])
            kwargs["transport"] = transport
            return real_client(**kwargs)

        with patch.object(target.httpx, "AsyncClient", side_effect=client):
            result = await target.download_http_url_bytes(
                URL, max_bytes=limit, strict_budget=budget
            )
        return result, calls

    async def test_success_counts_actual_entity_and_closes(self):
        budget = target.StrictHTTPBudget(
            max_bytes=READ + 32, max_requests=2, deadline_seconds=1
        )
        stream = Stream([b"abc"])
        result, calls = await self.request(budget, stream)
        self.assertEqual(result, b"abc")
        self.assertEqual((budget.response_bytes, budget.requests), (3, 1))
        self.assertEqual(len(calls), 1)
        self.assertTrue(stream.closed)

    async def test_reservation_refuses_before_any_request(self):
        budget = target.StrictHTTPBudget(
            max_bytes=READ + 15, max_requests=2, deadline_seconds=1
        )
        stream = Stream([b"abc"])
        with self.assertRaisesRegex(ValueError, "budget"):
            await self.request(budget, stream)
        self.assertEqual(budget.requests, 0)

    async def test_unknown_length_overflow_charges_reserved_entity_and_closes(self):
        budget = target.StrictHTTPBudget(
            max_bytes=2 * READ, max_requests=2, deadline_seconds=1
        )
        stream = Stream([b"x" * READ])
        with self.assertRaisesRegex(ValueError, "limit"):
            await self.request(budget, stream)
        self.assertTrue(stream.closed)
        self.assertEqual(budget.requests, 1)
        self.assertEqual(budget.response_bytes, READ + 16)
        self.assertTrue(budget.closed)

    async def test_gzip_refusal_precedes_body_iteration(self):
        budget = target.StrictHTTPBudget(
            max_bytes=2 * READ, max_requests=2, deadline_seconds=1
        )
        stream = Stream([b"compressed"])
        with self.assertRaisesRegex(ValueError, "encoding"):
            await self.request(budget, stream, headers={"content-encoding": "gzip"})
        self.assertTrue(stream.closed)
        self.assertTrue(budget.closed)

    async def test_redirect_is_terminal_and_has_one_request(self):
        budget = target.StrictHTTPBudget(
            max_bytes=2 * READ, max_requests=2, deadline_seconds=1
        )
        stream = Stream([b"redirect"])
        with self.assertRaisesRegex(ValueError, "status"):
            await self.request(
                budget, stream, status=302, headers={"location": URL + "/other"}
            )
        self.assertEqual(budget.requests, 1)
        self.assertTrue(stream.closed)

    async def test_401_has_zero_retry_and_terminal_budget(self):
        budget = target.StrictHTTPBudget(
            max_bytes=2 * READ, max_requests=2, deadline_seconds=1
        )
        stream = Stream([b"error"])
        with self.assertRaisesRegex(ValueError, "status"):
            await self.request(budget, stream, status=401)
        self.assertEqual(budget.requests, 1)
        self.assertTrue(budget.closed)
        self.assertTrue(stream.closed)

    async def test_absolute_deadline_stops_slow_trickle(self):
        budget = target.StrictHTTPBudget(
            max_bytes=2 * READ, max_requests=2, deadline_seconds=0.03
        )
        stream = Stream([b"a"] * 20, delay=0.01)
        with self.assertRaises(TimeoutError):
            await self.request(budget, stream)
        self.assertTrue(stream.closed)
        self.assertTrue(budget.closed)

    async def test_request_limit_precedes_second_dispatch(self):
        budget = target.StrictHTTPBudget(
            max_bytes=2 * READ, max_requests=1, deadline_seconds=1
        )
        await self.request(budget, Stream([b"a"]))
        with self.assertRaisesRegex(ValueError, "budget"):
            await self.request(budget, Stream([b"a"]))
        self.assertEqual(budget.requests, 1)

    async def test_declared_oversize_and_error_entity_close(self):
        for status in (200, 500):
            with self.subTest(status=status):
                budget = target.StrictHTTPBudget(
                    max_bytes=2 * READ, max_requests=2, deadline_seconds=1
                )
                stream = Stream([b"x" * 17])
                with self.assertRaisesRegex(ValueError, "limit"):
                    await self.request(
                        budget, stream, status=status, headers={"content-length": "17"}
                    )
                self.assertTrue(stream.closed)
                self.assertTrue(budget.closed)

    async def test_success_releases_headroom_and_third_request_refuses(self):
        budget = target.StrictHTTPBudget(
            max_bytes=READ + 19, max_requests=3, deadline_seconds=1
        )
        await self.request(budget, Stream([b"abc"]))
        await self.request(budget, Stream([b"abc"]))
        with self.assertRaisesRegex(ValueError, "budget"):
            await self.request(budget, Stream([b"abc"]))
        self.assertEqual((budget.response_bytes, budget.requests), (6, 2))
        self.assertTrue(budget.closed)

    async def test_cancellation_closes_response_and_budget(self):
        budget = target.StrictHTTPBudget(
            max_bytes=2 * READ, max_requests=2, deadline_seconds=1
        )
        stream = Stream([b"a"], delay=0.5)
        task = asyncio.create_task(self.request(budget, stream))
        await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(stream.closed)
        self.assertTrue(budget.closed)
        self.assertEqual(budget.requests, 1)

    async def test_shared_request_budget_serializes_concurrent_calls(self):
        budget = target.StrictHTTPBudget(
            max_bytes=2 * READ, max_requests=1, deadline_seconds=1
        )
        first = asyncio.create_task(self.request(budget, Stream([b"a"], delay=0.01)))
        await asyncio.sleep(0)
        second = asyncio.create_task(
            target.download_http_url_bytes(URL, max_bytes=16, strict_budget=budget)
        )
        results = await asyncio.gather(first, second, return_exceptions=True)
        self.assertEqual(results[0][0], b"a")
        self.assertIsInstance(results[1], ValueError)
        self.assertEqual((budget.response_bytes, budget.requests), (1, 1))

    async def test_malformed_content_length_refuses_before_body(self):
        for length in ("-1", "invalid", "١"):
            with self.subTest(length=length):
                budget = target.StrictHTTPBudget(
                    max_bytes=2 * READ, max_requests=2, deadline_seconds=1
                )
                stream = Stream([b"a"])
                with self.assertRaises((ValueError, UnicodeEncodeError)):
                    await self.request(
                        budget, stream, headers={"content-length": length}
                    )
                self.assertTrue(budget.closed)

    async def test_transport_read_bound_drift_refuses_before_dispatch(self):
        budget = target.StrictHTTPBudget(
            max_bytes=2 * READ, max_requests=2, deadline_seconds=1
        )
        with patch.object(target.AsyncHTTP11Connection, "READ_NUM_BYTES", READ + 1):
            with self.assertRaisesRegex(ValueError, "transport changed"):
                await self.request(budget, Stream([b"a"]))
        self.assertEqual(budget.requests, 0)

    async def test_real_http11_stream_stops_after_reserved_read(self):
        class NetworkStream:
            def __init__(self):
                self.pending = (
                    b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n" + b"x" * (2 * READ)
                )
                self.read_sizes = []
                self.received = 0
                self.closed = False

            async def read(self, max_bytes, timeout=None):
                self.read_sizes.append(max_bytes)
                result, self.pending = (
                    self.pending[:max_bytes],
                    self.pending[max_bytes:],
                )
                self.received += len(result)
                return result

            async def write(self, data, timeout=None):
                pass

            async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
                return self

            def get_extra_info(self, info):
                return None

            async def aclose(self):
                self.closed = True

        stream = NetworkStream()

        class Backend:
            async def connect_tcp(self, *args, **kwargs):
                return stream

        budget = target.StrictHTTPBudget(
            max_bytes=READ + 16, max_requests=1, deadline_seconds=1
        )
        with patch(
            "httpcore._async.connection_pool.AutoBackend", return_value=Backend()
        ):
            with self.assertRaisesRegex(ValueError, "limit"):
                await target.download_http_url_bytes(
                    URL, max_bytes=16, strict_budget=budget
                )
        self.assertEqual(stream.read_sizes, [READ])
        self.assertEqual(stream.received, READ)
        self.assertTrue(stream.closed)
        self.assertTrue(budget.closed)
        self.assertLessEqual(stream.received, budget.response_bytes)
        self.assertLessEqual(budget.response_bytes, budget.max_bytes)
