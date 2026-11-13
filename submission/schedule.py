"""Full-budget task planner; costs are tenths of output-equivalent tokens.

Ceiling each cost to a shared quantum gives at most 512 budget cells, including
zero. The dynamic program plans across all candidates; slots only limits dispatch.
Rounding can discard a feasible/better real-cost combination; neither estimated
probabilities nor costs guarantee actual contest outcomes. The caller must exclude
completed/in-flight tasks and reserve each request before sending.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TaskEstimate:
    task_id: str
    reward: float
    success_probability: float
    cost_units: int


def choose_tasks(candidates, budget_units, slots=5):
    """Plan the whole budget, then dispatch its first slots (density/free first).

    Input estimates are never mutated. Hidden expected answers and quality labels
    are not inputs. Zero-value candidates are omitted, even when they cost zero.
    """
    if type(budget_units) is not int or budget_units < 0:
        raise ValueError("budget_units must be a nonnegative integer")
    if type(slots) is not int or not 0 <= slots <= 5:
        raise ValueError("slots must be an integer from 0 to 5")
    tasks, ids, values = list(candidates), set(), []
    for task in tasks:
        if not isinstance(task, TaskEstimate):
            raise ValueError("candidates must contain TaskEstimate objects")
        if not isinstance(task.task_id, str) or not task.task_id.strip() or task.task_id in ids:
            raise ValueError("task_id must be nonempty and unique")
        ids.add(task.task_id)
        for value in (task.reward, task.success_probability):
            try:
                valid = type(value) in (int, float) and value >= 0 and math.isfinite(value)
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("reward and probability must be finite nonnegative numbers")
        if task.success_probability > 1:
            raise ValueError("success_probability must be at most one")
        if type(task.cost_units) is not int or task.cost_units < 0:
            raise ValueError("cost_units must be a nonnegative integer")
        values.append(task.reward * task.success_probability)
    if slots == 0 or not tasks:
        return []

    quantum = max(1, (budget_units + 510) // 511)
    capacity = budget_units // quantum
    states = [None] * (capacity + 1)
    states[0] = (0.0, 0, ())  # Expected reward, exact estimated cost, task indices.
    for index, task in enumerate(tasks):
        weight = (task.cost_units + quantum - 1) // quantum
        if values[index] == 0 or weight > capacity:
            continue
        # Descending costs prevent reuse; a zero-cost task updates each cell once.
        for cost in range(capacity, weight - 1, -1):
            previous = states[cost - weight]
            if previous is None:
                continue
            value = previous[0] + values[index]
            exact_cost = previous[1] + task.cost_units
            current = states[cost]
            if current is None or (value, -exact_cost) > (current[0], -current[1]):
                states[cost] = (value, exact_cost, previous[2] + (index,))

    best = max((state for state in states if state is not None),
               key=lambda state: (state[0], -state[1], -len(state[2])))
    chosen = [tasks[index] for index in best[2]]
    return sorted(chosen, key=lambda task: (
        task.cost_units != 0,
        -(task.reward * task.success_probability / task.cost_units
          if task.cost_units else task.reward * task.success_probability),
        task.cost_units, task.task_id))[:slots]
