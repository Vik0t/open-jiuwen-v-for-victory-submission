"""One native openJiuwen request per answer; no tools, retries or reasoning edits.

One Gateway context owns Runner. answer returns task_id/status/text/model/usage/
cost_units; text is only the Runner's final answer, even if usage is unknown.
"""
import asyncio
from contextvars import ContextVar
import json
import logging
import os
from uuid import uuid4

import httpx
from openai import AsyncOpenAI
from openjiuwen.core.common.logging import LogManager
from openjiuwen.core.common.logging.log_config import log_config
from openjiuwen.core.foundation.llm import ModelClientConfig, ModelRequestConfig
from openjiuwen.core.foundation.llm.model_clients.openai_model_client import OpenAIModelClient
from openjiuwen.core.runner import Runner
from openjiuwen.core.runner.runner_config import RunnerConfig
from openjiuwen.core.single_agent import AgentCard, ReActAgent, ReActAgentConfig

MODELS = frozenset({"qwen3.8-27b", "qwen3.8-27b-noreason", "qwen3.6-35b-a3b", "qwen3.6-35b-a3b-noreason"})
ACTIVE = ContextVar("contest_gateway_request", default=None)


def _failed_before_http(events):
    """Only complete, single-connection TCP/TLS failure traces are evidence."""
    tcp_start = "connection.connect_tcp.started"
    tcp_done = "connection.connect_tcp.complete"
    tcp_failed = "connection.connect_tcp.failed"
    tls_start = "connection.start_tls.started"
    tls_failed = "connection.start_tls.failed"
    return events == [tcp_start, tcp_failed] or events == [tcp_start, tcp_done, tls_start, tls_failed]


class Gateway:
    def __init__(self, budget, base_url, api_key, model=None, local_address=None):
        self.model = os.environ.get("LLM_MODEL", "qwen3.8-27b-noreason") if model is None else model
        if not isinstance(self.model, str) or self.model.casefold() not in MODELS:
            raise ValueError("Model is outside the four permitted contest IDs")
        self.budget, self._base_url, self._api_key = budget, base_url, api_key
        self._local_address, self._client = local_address, None
        self._semaphore = asyncio.Semaphore(3)

    async def __aenter__(self):
        if self._client is not None:
            raise RuntimeError("Gateway is already open")
        # Caller changes cwd before importing SDK; do not log provider bodies/errors.
        config = log_config.get_snapshot()
        config.update(level="CRITICAL", loggers={})
        log_config.load_from_dict(config)  # Also covers SDK loggers created later.
        for logger in LogManager.get_all_loggers().values():
            logger.set_level(logging.CRITICAL)
        transport = httpx.AsyncHTTPTransport(local_address=self._local_address, retries=0,
            limits=httpx.Limits(max_connections=3, max_keepalive_connections=3))
        self._http = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(180, connect=10, read=600),
            trust_env=False, event_hooks={"request": [self._request], "response": [self._response]})
        self._client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key, http_client=self._http, max_retries=0)
        self._provider = "contest_" + uuid4().hex
        client, name = self._client, self._provider
        class ObservedOpenAI(OpenAIModelClient):
            __client_name__, _client_cache = name, {}
            def _build_async_openai_client(self, timeout=None):
                return client
        Runner.set_config(RunnerConfig(distributed_mode=False))
        try:
            await Runner.start()
        except BaseException:
            await self._client.close()
            self._client = None
            raise
        return self

    async def __aexit__(self, *exc):
        try:
            await Runner.stop()
        finally:
            await self._client.close()
            self._client = None

    async def _request(self, request):
        if not request.url.path.endswith("/chat/completions"):
            raise RuntimeError("Unexpected model endpoint")
        state = ACTIVE.get()
        if state is None or state["attempted"]:
            raise RuntimeError("Expected one model request per answer")
        body = await request.aread()
        payload = json.loads(body)
        if payload.get("model") != self.model or type(payload.get("max_tokens")) is not int or payload["max_tokens"] <= 0:
            raise ValueError("Unexpected model or missing output cap")
        state["input_bound"] = len(body) + 256
        ticket = self.budget.reserve(state["input_bound"], payload["max_tokens"])
        if ticket is None:
            state["status"] = "budget_unavailable"
            raise RuntimeError("Local budget reservation denied")
        state.update(ticket=ticket, attempted=True)
        request.extensions.update(contest_ticket=ticket, contest_state=state)
        # A missing or pre-existing trace is ambiguous: retain the reservation.
        if "trace" not in request.extensions:
            events = []
            state["connect_trace"] = events
            async def trace(name, info):
                events.append(name)
            request.extensions["trace"] = trace
        # From this point any transport failure is conservatively unknown, not free.

    async def _response(self, response):
        state = response.request.extensions["contest_state"]
        ticket = response.request.extensions["contest_ticket"]
        try:
            data = json.loads(await response.aread())
        except (ValueError, UnicodeError):
            data = None
        state["cost_units"] = self.budget.settle(ticket, data)
        state["settled"] = True
        usage = data.get("usage") if isinstance(data, dict) else None
        if isinstance(usage, dict):
            state["usage"] = {k: v if type(v) is int and v >= 0 else None
                              for k in ("prompt_tokens", "completion_tokens", "total_tokens") for v in [usage.get(k)]}
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict) and type(details.get("reasoning_tokens")) is int and details["reasoning_tokens"] >= 0:
                state["usage"]["completion_tokens_details"] = {"reasoning_tokens": details["reasoning_tokens"]}
        if state["cost_units"] is None:
            state["status"] = "unknown_usage"

    async def answer(self, task_id, prompt, max_output=512, system_prompt=None, json_mode=False):
        if self._client is None:
            raise RuntimeError("Use Gateway as an async context manager")
        if not isinstance(prompt, str) or type(max_output) is not int or max_output <= 0:
            raise ValueError("Require text and a positive integer output cap")
        async with self._semaphore:
            state = dict(ticket=None, attempted=False, settled=False, usage=None, cost_units=None, status="ok")
            token = ACTIVE.set(state)
            text, error = None, None
            try:
                agent = ReActAgent(AgentCard(id="contest_" + uuid4().hex, name="contest_solver", description="Task solver"))
                agent.configure(ReActAgentConfig(model_name=self.model, model_provider=self._provider,
                    model_client_config=ModelClientConfig(client_provider=self._provider, endpoint_profile="openai_compatible",
                        api_key=self._api_key, api_base=self._base_url, timeout=600, max_retries=0),
                    model_config_obj=ModelRequestConfig(model=self.model, temperature=0, max_tokens=max_output,
                        **({"extra_body": {"response_format": {"type": "json_object"}}} if json_mode else {})),
                    prompt_template=[{"role": "system", "content": system_prompt or "Solve the task. Return only the final answer in the requested format."}],
                    max_iterations=1, max_iterations_hard_limit=1, on_max_iterations="return_observations", parallel_tool_calls=False))
                result = await Runner.run_agent(agent, {"query": prompt, "conversation_id": agent.card.id})
                if result.get("result_type") == "answer" and isinstance(result.get("output"), str):
                    text = result["output"]
                elif state["status"] == "ok":
                    state["status"] = "agent_error"
            except Exception as exc:
                error = type(exc).__name__  # Never return provider exception text/headers.
                if state["status"] == "ok":
                    state["status"] = "agent_error"
            finally:
                if state["ticket"] is not None and not state["settled"]:
                    if _failed_before_http(state.get("connect_trace")):
                        self.budget.release(state["ticket"])
                        state["status"] = "not_sent"
                    else:
                        self.budget.settle(state["ticket"], None)
                        state["status"] = "unknown_usage"
                ACTIVE.reset(token)
            return {"task_id": task_id, "status": state["status"], "text": text, "model": self.model,
                    "usage": state["usage"], "cost_units": state["cost_units"], "error": error,
                    "input_bound": state.get("input_bound")}
