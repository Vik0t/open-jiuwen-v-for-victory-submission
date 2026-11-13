from itertools import combinations
import random

import pytest

from submission.schedule import TaskEstimate, choose_tasks


def test_knapsack_avoids_greedy_density_counterexample():
    tasks = [TaskEstimate("a", 60, 1, 51), TaskEstimate("b", 50, 1, 50),
             TaskEstimate("c", 50, 1, 50)]
    assert [task.task_id for task in choose_tasks(tasks, 100)] == ["b", "c"]


def test_full_plan_prefers_twenty_cheap_tasks_to_five_expensive_ones():
    expensive = [TaskEstimate(f"expensive{index}", 30, 1, 20) for index in range(5)]
    cheap = [TaskEstimate(f"cheap{index:02}", 10, 1, 5) for index in range(20)]
    pending, budget, finished = expensive + cheap, 100, []
    first = choose_tasks(pending, budget, slots=5)
    assert first == cheap[:5]
    while batch := choose_tasks(pending, budget, slots=5):
        budget -= sum(task.cost_units for task in batch)
        finished.extend(batch)
        pending = [task for task in pending if task not in batch]
    assert finished == cheap
    assert budget == 0
    assert sum(task.reward for task in finished) == 200


def test_one_hundred_free_candidates_are_unique_bounded_and_ranked():
    tasks = [TaskEstimate(str(index), index, 1, 0) for index in range(100)]
    chosen = choose_tasks(tasks, 0)
    assert [task.task_id for task in chosen] == [str(index) for index in range(99, 94, -1)]
    assert len(set(task.task_id for task in chosen)) == 5
    assert choose_tasks([TaskEstimate("no_chance", 100, 0, 0)], 0) == []
    assert choose_tasks([TaskEstimate("zero_reward", 0, 1, 0)], 0) == []


def test_budget_units_slots_and_order_are_explicit():
    tasks = [TaskEstimate("low", 12, 0.5, 6), TaskEstimate("high", 10, 1, 4),
             TaskEstimate("free", 1, 1, 0)]
    assert choose_tasks(tasks, 3) == [tasks[2]]
    assert [task.task_id for task in choose_tasks(tasks, 10)] == ["free", "high", "low"]
    assert choose_tasks(tasks, 10, slots=1) == [tasks[2]]
    assert choose_tasks(tasks, 10, slots=0) == []
    assert choose_tasks([], 10) == []
    assert choose_tasks([TaskEstimate("needs11", 10, 1, 11)], 10) == []


def test_unrounded_small_full_plans_match_exhaustive_expected_value():
    rng = random.Random(42)
    for _ in range(20):
        tasks = [TaskEstimate(str(index), rng.randint(0, 20), rng.choice([0, 0.25, 0.5, 1]),
                              rng.randint(0, 15)) for index in range(5)]
        budget = rng.randint(0, 30)
        feasible = [subset for count in range(len(tasks) + 1) for subset in combinations(tasks, count)
                    if sum(task.cost_units for task in subset) <= budget]
        optimum = max(sum(task.reward * task.success_probability for task in subset) for subset in feasible)
        chosen = choose_tasks(tasks, budget)
        assert sum(task.reward * task.success_probability for task in chosen) == optimum


def test_conservative_rounding_never_exceeds_estimates_for_one_hundred_tasks():
    rng = random.Random(137)
    tasks = [TaskEstimate(str(index), rng.uniform(1, 500), rng.random(), rng.randint(0, 100000))
             for index in range(100)]
    original = list(tasks)
    for budget in (0, 1, 512, 12345, 1000000):
        for slots in range(6):
            chosen = choose_tasks(tasks, budget, slots)
            assert len(chosen) <= slots
            assert len({task.task_id for task in chosen}) == len(chosen)
            assert sum(task.cost_units for task in chosen) <= budget
    assert tasks == original
    # A feasible true-cost combination can be rejected by conservative rounding.
    awkward = [TaskEstimate("a", 10, 1, 500), TaskEstimate("b", 10, 1, 523)]
    assert len(choose_tasks(awkward, 1023)) == 1


@pytest.mark.parametrize("budget,slots", [(-1, 5), (True, 5), (1.0, 5), ("10", 5),
    (10, -1), (10, 6), (10, True), (10, 1.0)])
def test_invalid_limits_are_rejected(budget, slots):
    with pytest.raises(ValueError):
        choose_tasks([], budget, slots)


@pytest.mark.parametrize("task", [TaskEstimate("", 1, 1, 1), TaskEstimate(" ", 1, 1, 1),
    TaskEstimate("a", -1, 1, 1), TaskEstimate("a", float("nan"), 1, 1),
    TaskEstimate("a", float("inf"), 1, 1), TaskEstimate("a", True, 1, 1),
    TaskEstimate("a", "1", 1, 1), TaskEstimate("a", 1, -0.1, 1),
    TaskEstimate("a", 1, 1.1, 1), TaskEstimate("a", 1, float("nan"), 1),
    TaskEstimate("a", 1, True, 1), TaskEstimate("a", 1, 1, -1),
    TaskEstimate("a", 1, 1, True), TaskEstimate("a", 1, 1, 1.5), {"expected": "hidden"}])
def test_invalid_estimates_are_rejected_even_with_zero_slots(task):
    with pytest.raises(ValueError):
        choose_tasks([task], 100, slots=0)


def test_duplicate_task_ids_are_rejected():
    task = TaskEstimate("same", 1, 1, 1)
    with pytest.raises(ValueError):
        choose_tasks([task, task], 10)
