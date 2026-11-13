"""Native SDK + MockTransport; all network/model calls are local simulations."""
import asyncio
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SDK = all(importlib.util.find_spec(name) is not None for name in ("openjiuwen", "openai", "httpx"))
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(SDK, "Requires the existing openJiuwen SDK interpreter")
class GatewayChecks(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="contest-gateway-sdk-")
        cls.cwd = contextlib.chdir(cls.temporary.name)
        cls.cwd.__enter__()  # SDK's import-time logs belong in /tmp.
        sys.path.insert(0, str(ROOT))
        import httpx
        from submission.budget import Budget
        from submission.gateway import Gateway
        cls.httpx, cls.Budget, cls.Gateway = httpx, Budget, Gateway

    @classmethod
    def tearDownClass(cls):
        cls.cwd.__exit__(None, None, None)
        cls.temporary.cleanup()

    def response(self, request, text="FINAL", usage=True):
        counts = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                  "completion_tokens_details": {"reasoning_tokens": 7}, "api_key": "DO_NOT_EXPORT"} if usage is True else usage
        body = {"id": "mock", "object": "chat.completion", "created": 0, "model": "Qwen3.8-27B",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text,
                            "reasoning_content": "PRIVATE_REASONING_DO_NOT_RETURN"}}], "usage": counts}
        if json.loads(request.content).get("stream"):
            common = {"id": "mock", "object": "chat.completion.chunk", "created": 0, "model": "Qwen3.8-27B"}
            events = [
                {**common, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
                {**common, "choices": [{"index": 0, "delta": {"content": text, "reasoning_content": "PRIVATE_REASONING_DO_NOT_RETURN"}, "finish_reason": None}]},
                {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            if counts is not None:
                events.append({**common, "choices": [{"index": 0, "delta": {}, "finish_reason": None}], "usage": counts})
            data = ("".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n").encode()
            class Stream(self.httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield data
                async def aclose(self):
                    pass
            return self.httpx.Response(200, stream=Stream(), headers={"Content-Type": "text/event-stream"}, request=request)
        return self.httpx.Response(200, json=body, request=request)

    def gateway(self, budget):
        return self.Gateway(budget, "https://unit.invalid/v1", "TEST_KEY_NOT_REAL", model="Qwen3.8-27B")

    async def test_actual_runner_final_text_usage_and_exact_requested_model(self):
        budget, requests = self.Budget(10000), []
        def handler(request):
            requests.append(request)
            body = json.loads(request.content)
            assert body["model"] == "Qwen3.8-27B" and body["max_tokens"] == 64
            assert "chat_template_kwargs" not in body and "reasoning_effort" not in body
            assert budget.reserved_units == len(request.content) + 256 + 640
            assert isinstance(request.extensions["contest_ticket"], int)
            return self.response(request)
        with patch("submission.gateway.httpx.AsyncHTTPTransport", return_value=self.httpx.MockTransport(handler)):
            async with self.gateway(budget) as gateway:
                result = await gateway.answer("task-1", "Return FINAL", max_output=64)
        self.assertEqual((result["status"], result["text"], result["cost_units"]), ("ok", "FINAL", 300))
        self.assertEqual((len(requests), budget.spent_units, budget.reserved_units), (1, 300, 0))
        for secret in ["PRIVATE_REASONING", "DO_NOT_EXPORT", "TEST_KEY_NOT_REAL"]:
            self.assertNotIn(secret, json.dumps(result))
            self.assertFalse(any(secret in path.read_text(errors="replace")
                                 for path in Path(self.temporary.name).rglob("*.log")))

    async def test_budget_denial_happens_before_any_transport_request(self):
        budget, calls = self.Budget(1), []
        def handler(request):
            calls.append(request)
            return self.response(request)
        with patch("submission.gateway.httpx.AsyncHTTPTransport", return_value=self.httpx.MockTransport(handler)):
            async with self.gateway(budget) as gateway:
                result = await gateway.answer("tiny", "No budget")
        self.assertEqual(result["status"], "budget_unavailable")
        self.assertEqual((calls, budget.spent_units, budget.reserved_units), ([], 0, 0))

    async def test_unknown_usage_freezes_and_cannot_leak_to_the_next_answer(self):
        budget, calls = self.Budget(10000), []
        def handler(request):
            calls.append(request)
            return self.response(request, usage=None)
        with patch("submission.gateway.httpx.AsyncHTTPTransport", return_value=self.httpx.MockTransport(handler)):
            async with self.gateway(budget) as gateway:
                first = await gateway.answer("unknown", "First")
                second = await gateway.answer("blocked", "Second")
        self.assertEqual(first["status"], "unknown_usage")
        self.assertIsNone(first["cost_units"])
        self.assertTrue(budget.frozen and budget.reserved_units > 0)
        self.assertEqual((len(calls), second["status"]), (1, "budget_unavailable"))

    async def test_network_timeout_has_one_attempt_and_unknown_cost(self):
        budget, calls = self.Budget(10000), []
        def handler(request):
            calls.append(request)
            raise self.httpx.ReadTimeout("SENSITIVE_PROVIDER_ERROR", request=request)
        with patch("submission.gateway.httpx.AsyncHTTPTransport", return_value=self.httpx.MockTransport(handler)):
            async with self.gateway(budget) as gateway:
                result = await gateway.answer("timeout", "First")
        self.assertEqual((len(calls), result["status"]), (1, "unknown_usage"))
        self.assertIsNone(result["text"])
        self.assertTrue(budget.frozen and budget.reserved_units > 0)
        self.assertNotIn("SENSITIVE_PROVIDER_ERROR", json.dumps(result))

    async def test_parallel_native_answers_keep_their_own_ticket_and_usage(self):
        budget, active, peak, states = self.Budget(100000), 0, 0, set()
        async def handler(request):
            nonlocal active, peak
            state = request.extensions["contest_state"]
            states.add(id(state))
            index = int(json.loads(request.content)["messages"][-1]["content"][-1])
            active += 1
            peak = max(peak, active)
            await asyncio.sleep((7 - index) * 0.005)
            active -= 1
            return self.response(request, f"FINAL-{index}", {"prompt_tokens": 100 + index,
                "completion_tokens": 20 + index, "total_tokens": 120 + 2 * index})
        with patch("submission.gateway.httpx.AsyncHTTPTransport", return_value=self.httpx.MockTransport(handler)):
            async with self.gateway(budget) as gateway:
                results = await asyncio.gather(*(gateway.answer(str(i), f"job{i}", 64) for i in range(7)))
        self.assertEqual([r["text"] for r in results], [f"FINAL-{i}" for i in range(7)])
        self.assertEqual([r["cost_units"] for r in results], [300 + 11 * i for i in range(7)])
        self.assertEqual((peak, len(states), budget.spent_units, budget.reserved_units), (3, 7, 2331, 0))

    async def test_cancellation_after_request_dispatch_never_releases_unknown_cost(self):
        budget, started = self.Budget(10000), asyncio.Event()
        async def handler(request):
            started.set()
            await asyncio.Future()
        with patch("submission.gateway.httpx.AsyncHTTPTransport", return_value=self.httpx.MockTransport(handler)):
            async with self.gateway(budget) as gateway:
                task = asyncio.create_task(gateway.answer("cancel", "Wait"))
                await asyncio.wait_for(started.wait(), 5)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        self.assertTrue(budget.frozen and budget.reserved_units > 0)

    def test_model_whitelist_is_casefold_validation_without_flash_alias(self):
        from submission.gateway import MODELS
        for model in MODELS:
            gateway = self.Gateway(self.Budget(1), "https://unit.invalid/v1", "test", model.upper())
            self.assertEqual(gateway.model, model.upper())
        with self.assertRaises(ValueError):
            self.Gateway(self.Budget(1), "https://unit.invalid/v1", "test", "qwen3.8-flash-next")
