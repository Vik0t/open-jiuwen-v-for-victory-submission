from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest

from submission.budget import Budget, normalize_actual_usage


def test_ten_input_tokens_equal_one_output_and_env_is_exact(monkeypatch):
    monkeypatch.setenv("BUDGET", "1.5")
    budget = Budget()
    assert budget.limit_units == 15
    assert normalize_actual_usage({"prompt_tokens": 10, "completion_tokens": 0}) == 10
    assert normalize_actual_usage({"prompt_tokens": 0, "completion_tokens": 1}) == 10
    ticket = budget.reserve(5, 1)
    assert ticket is not None and budget.remaining == 0
    assert budget.settle(ticket, {"usage": {"prompt_tokens": 5, "completion_tokens": 1}}) == 15
    assert budget.spent_units == 15 and budget.reserved_units == 0
    assert not budget.afford(0, 1)


def test_reasoning_details_are_not_added_twice_and_total_resolves_extra_output():
    usage = {"prompt_tokens": 100, "completion_tokens": 20,
             "completion_tokens_details": {"reasoning_tokens": 7}}
    assert normalize_actual_usage(usage) == 300
    assert normalize_actual_usage({**usage, "total_tokens": 120}) == 300
    assert normalize_actual_usage({**usage, "total_tokens": 130}) == 400
    assert normalize_actual_usage({"prompt_tokens": 100, "total_tokens": 130}) == 400
    assert normalize_actual_usage({**usage, "reasoning_tokens": 7}) is None
    assert normalize_actual_usage({**usage, "reasoning_tokens": 7, "total_tokens": 130}) == 400


def test_unknown_usage_freezes_without_releasing_or_fabricating_cost():
    budget = Budget(100)
    sent, unsent = budget.reserve(100, 10), budget.reserve(100, 10)
    assert budget.settle(sent, None) is None
    assert budget.frozen and budget.reason == "unknown_usage"
    assert budget.spent_units == 0  # Known subtotal, not an exact whole bill.
    assert budget.reserved_units == 400 and budget.remaining_units == 0
    assert budget.reserve(0, 1) is None and not budget.afford(0, 0)
    with pytest.raises(ValueError):
        budget.release(sent)
    assert budget.release(unsent) == 200
    assert budget.reserved_units == 200 and budget.frozen


def test_parallel_reservations_cannot_allocate_the_same_budget_twice():
    budget = Budget(80, max_inflight=5)
    with ThreadPoolExecutor(max_workers=8) as pool:
        tickets = list(pool.map(lambda _: budget.reserve(10, 19), range(32)))
    allocated = [ticket for ticket in tickets if ticket is not None]
    assert len(allocated) == len(set(allocated)) == 4
    assert budget.reserved_units == budget.limit_units == 800
    assert budget.settle(allocated[0], {"prompt_tokens": 10, "completion_tokens": 9}) == 100
    assert budget.remaining_units == 100
    assert budget.release(allocated[1]) == 200
    assert budget.reserve(0, 30) is not None
    assert budget.spent_units + budget.reserved_units == budget.limit_units
    assert budget.reserve(1, 0) is None


@pytest.mark.parametrize("configured, expected", [(None, 3), (5, 5)])
def test_inflight_limit_applies_even_with_unallocated_budget(configured, expected):
    budget = Budget(10000) if configured is None else Budget(10000, max_inflight=configured)
    tickets = [budget.reserve(0, 1) for _ in range(expected)]
    assert all(ticket is not None for ticket in tickets) and budget.remaining_units > 0
    assert budget.reserve(0, 1) is None
    budget.release(tickets[0])
    assert budget.reserve(0, 1) is not None


@pytest.mark.parametrize("prompt, output", [(11, 0), (0, 11)])
def test_individual_request_bounds_cannot_cancel_each_other(prompt, output):
    budget = Budget(1000)
    ticket = budget.reserve(10, 10)
    actual = prompt + 10 * output
    assert actual <= 110  # A lower total does not excuse exceeding one bound.
    assert budget.settle(ticket, {"prompt_tokens": prompt, "completion_tokens": output}) == actual
    assert budget.spent_units == actual and budget.reserved_units == 0
    assert budget.reason == "request_bound_exceeded" and budget.remaining_units == 0
    assert budget.reserve(0, 1) is None


def test_missing_environment_budget_is_not_guessed(monkeypatch):
    monkeypatch.delenv("BUDGET", raising=False)
    with pytest.raises(ValueError):
        Budget()


def test_actual_overrun_blocks_new_requests_but_existing_usage_can_settle():
    budget = Budget(10)
    first, second = budget.reserve(0, 5), budget.reserve(0, 5)
    assert budget.settle(first, {"prompt_tokens": 1, "completion_tokens": 5}) == 51
    assert budget.frozen and budget.reason == "budget_overcommitted"
    assert budget.settle(second, {"prompt_tokens": 0, "completion_tokens": 1}) == 10
    assert budget.spent_units == 61 and budget.reserved_units == 0 and budget.remaining == 0
    with pytest.raises(ValueError):
        budget.settle(first, {"prompt_tokens": 0, "completion_tokens": 1})


@pytest.mark.parametrize("usage", [None, {}, {"usage": None}, {"prompt_tokens": 10},
    {"input_tokens": 10, "output_tokens": 2}, {"prompt_tokens": True, "completion_tokens": 1},
    {"prompt_tokens": 10, "completion_tokens": -1}, {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 11},
    {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": "12"},
    {"prompt_tokens": 10, "completion_tokens": 2, "completion_tokens_details": {"reasoning_tokens": 3}}])
def test_unknown_or_inconsistent_formats_are_unsupported(usage):
    assert normalize_actual_usage(usage) is None


@pytest.mark.parametrize("limit", [True, -1, 0.1, "NaN", "Infinity", "0.01"])
def test_invalid_budget_never_creates_an_allocation(limit):
    with pytest.raises(ValueError):
        Budget(limit)


def test_strict_ticket_and_count_validation():
    budget = Budget(Decimal("100000000000000000000000000001.1"))
    assert budget.limit_units == 1000000000000000000000000000011
    with pytest.raises(ValueError):
        budget.reserve(True, 1)
    ticket = budget.reserve(1, 1)
    with pytest.raises(ValueError):
        budget.release(True)
    assert budget.release(ticket) == 11
