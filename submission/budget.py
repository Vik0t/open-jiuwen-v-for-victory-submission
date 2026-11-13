"""Local guard, not the proxy's authority. Units are tenths of output tokens.

OpenAI-compatible completion_tokens includes completion_tokens_details.reasoning_tokens.
Other standalone reasoning counters need a consistent total; their meaning is not guessed.
"""
from decimal import Decimal, InvalidOperation
import os
from threading import Lock


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _cost(input_tokens, output_tokens):
    if _count(input_tokens) is None or _count(output_tokens) is None:
        raise ValueError("Token counts must be nonnegative integers")
    return input_tokens + 10 * output_tokens


def _usage_counts(response):
    """Return (input, billed output), or None for unsupported usage."""
    if not isinstance(response, dict):
        return None
    usage = response.get("usage", response)
    if not isinstance(usage, dict):
        return None
    prompt, completion, total = (usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
    if _count(prompt) is None or any(v is not None and _count(v) is None for v in (completion, total)):
        return None
    if completion is None and total is None:
        return None
    if total is not None and total < prompt + (completion or 0):
        return None
    output = total - prompt if total is not None else completion
    details = usage.get("completion_tokens_details")
    if details is not None and not isinstance(details, dict):
        return None
    reasoning = (details or {}).get("reasoning_tokens")
    if reasoning is not None and (_count(reasoning) is None or reasoning > output):
        return None
    for name in ("reasoning_tokens", "thinking_tokens"):
        separate = usage.get(name)
        if separate is not None and (_count(separate) is None or separate > output or (separate > 0 and total is None)):
            return None
    return prompt, output


def normalize_actual_usage(response):
    """Return I+10*O, or None for unknown/inconsistent/unsupported usage."""
    counts = _usage_counts(response)
    return None if counts is None else _cost(*counts)


class Budget:
    """Reserve before sending; settle every sent request; release only unsent ones.

    limit (or env BUDGET) is output-equivalent tokens, without float rounding.
    *_units and remaining are integer tenths. spent_units is the known subtotal;
    unknown usage retains its reservation and permanently blocks new allocations.
    The runner controls waiting/dispatch; this guard admits at most five in flight.
    """

    def __init__(self, limit=None, max_inflight=3):
        value = os.environ.get("BUDGET") if limit is None else limit
        if isinstance(value, bool) or not isinstance(value, (int, str, Decimal)):
            raise ValueError("Supply a budget or set BUDGET; floats are unsupported")
        try:
            amount = Decimal(value)
        except InvalidOperation as error:
            raise ValueError("Invalid BUDGET") from error
        if not amount.is_finite() or amount < 0:
            raise ValueError("BUDGET must be finite and nonnegative")
        numerator, denominator = amount.as_integer_ratio()
        if numerator * 10 % denominator or type(max_inflight) is not int or not 1 <= max_inflight <= 5:
            raise ValueError("Require whole tenths and 1..5 in-flight requests")
        self.limit_units = numerator * 10 // denominator
        self.max_inflight, self.spent_units, self.reason = max_inflight, 0, None
        self._pending, self._unknown, self._next = {}, set(), 0
        self._bounds = {}
        self._lock = Lock()

    @property
    def frozen(self):
        return self.reason is not None

    @property
    def reserved_units(self):
        with self._lock:
            return sum(self._pending.values())

    @property
    def remaining_units(self):
        """Usable local authorization, not an authoritative proxy balance."""
        with self._lock:
            return 0 if self.frozen else max(0, self.limit_units - self.spent_units - sum(self._pending.values()))

    @property
    def remaining(self):
        return self.remaining_units

    def afford(self, estimated_input, max_output):
        """Advisory only: a successful reserve ticket is still mandatory."""
        amount = _cost(estimated_input, max_output)
        with self._lock:
            return not self.frozen and len(self._pending) < self.max_inflight and self.spent_units + sum(self._pending.values()) + amount <= self.limit_units

    def reserve(self, estimated_input, max_output):
        amount = _cost(estimated_input, max_output)
        with self._lock:
            if self.frozen or len(self._pending) >= self.max_inflight or self.spent_units + sum(self._pending.values()) + amount > self.limit_units:
                return None
            self._next += 1
            self._pending[self._next] = amount
            self._bounds[self._next] = (estimated_input, max_output)
            return self._next

    def release(self, ticket):
        """Cancel an unsent request, never a timeout/unknown billed request."""
        with self._lock:
            if type(ticket) is not int or ticket not in self._pending or ticket in self._unknown:
                raise ValueError("Unknown reservation or unresolved usage")
            self._bounds.pop(ticket)
            return self._pending.pop(ticket)

    def settle(self, ticket, response):
        counts = _usage_counts(response)
        amount = None if counts is None else _cost(*counts)
        with self._lock:
            if type(ticket) is not int or ticket not in self._pending:
                raise ValueError("Unknown reservation")
            if amount is None:
                self._unknown.add(ticket)
                self.reason = "unknown_usage"
                return None
            self._pending.pop(ticket)
            input_bound, output_cap = self._bounds.pop(ticket)
            self._unknown.discard(ticket)
            self.spent_units += amount
            if self.spent_units + sum(self._pending.values()) > self.limit_units:
                self.reason = self.reason or "budget_overcommitted"
            elif counts[0] > input_bound or counts[1] > output_cap:
                self.reason = self.reason or "request_bound_exceeded"
            return amount
