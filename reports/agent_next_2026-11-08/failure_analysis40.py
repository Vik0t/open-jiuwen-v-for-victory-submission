"""Summarize completed forty-task runs. No model calls or code execution."""
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / 'experiments/agent_next_2026-11-08'
DATA = ROOT / 'experiments/agent_next_holdout_2026-11-08'
POLICIES = ['v3_blind40', 'v4_blind40', 'v3_blind40_low', 'v4_blind40_low']
TRUTH = json.loads((DATA / 'private/answers.json').read_text())


def read(path):
    return json.loads(path.read_text())


def action(text):
    text = (text or '').strip()
    if text.startswith('```json\n') and text.endswith('\n```'):
        text = text[8:-4]
    try:
        value = json.loads(text)
    except ValueError:
        return None
    allowed = {'python', 'read', 'edit', 'write', 'answer', 'abstain'}
    return value if isinstance(value, dict) and value.get('action') in allowed else None


def candidate(row):
    for event in row.get('steps', []):
        observation = event.get('observation')
        if not isinstance(observation, dict) or not observation.get('checkpoint_saved'):
            continue
        if 'answer_proposed' in observation:
            return str(observation['answer_proposed']).strip()
        for line in observation.get('stdout', '').splitlines():
            if line.startswith('__AGENT_ANSWER__'):
                return str(json.loads(line.removeprefix('__AGENT_ANSWER__'))).strip()
    return None


def correct(task_id, answer, check):
    expected = TRUTH[task_id]['answer']
    if answer is None:
        return False
    if check == 'exact_int':
        try:
            return int(answer) == int(expected)
        except ValueError:
            return False
    return answer == expected


def main():
    output = {'scope': 'Completed forty-task runs only; fresh twenty tasks not read',
              'cost_unit': 'input + 10*output; divide by 10 for reported output equivalents',
              'runs': [], 'noncode_reviews': [], 'code_reviews': [], 'source_hashes': {}}
    passed_sets = []
    for policy in POLICIES:
        for run_name in ['r1', 'r2']:
            folder = EXPERIMENTS / policy / 'runs' / run_name
            scores, rows = read(folder / 'score.json'), read(folder / 'task_results.json')
            calls = [json.loads(line) for line in (folder / 'calls.jsonl').read_text().splitlines()]
            score_map = {row['task_id']: row for row in scores['by_task']}
            tasks = {path.parent.name: read(path) for path in (EXPERIMENTS / policy / 'tasks').glob('*/task.json')}
            stage_cost, stage_calls, invalid_cost, invalid_calls = Counter(), Counter(), Counter(), Counter()
            prior, task_cost = {}, Counter()
            flattened = []
            for row in rows:
                task_id, stage = row['task_id'], row.get('stage', 'main')
                events = row.get('steps', [])
                stage_cost[stage] += sum(event.get('cost_units') or 0 for event in events)
                task_cost[task_id] += sum(event.get('cost_units') or 0 for event in events)
                stage_calls[stage] += len(events)
                for event in events:
                    flattened.append((stage, event))
                    if event.get('status') == 'ok' and action(event.get('text')) is None:
                        invalid_calls[stage] += 1
                        invalid_cost[stage] += event.get('cost_units') or 0
                if stage == 'review':
                    saved = prior.get(task_id)
                    result = dict(policy=policy, run=run_name, task_id=task_id,
                                  type=tasks[task_id]['task_type'], reward=tasks[task_id]['reward'],
                                  actual_cost_units=sum(event.get('cost_units') or 0 for event in events),
                                  model_steps=len(events), source_changes=row.get('progress', {}).get('source_changes'),
                                  review_status=row.get('review_status'), final_passed=score_map[task_id]['passed'])
                    if tasks[task_id]['task_type'] == 'coding':
                        result['initial_quality_inferable_from_unchanged_source'] = result['source_changes'] == 0
                        output['code_reviews'].append(result)
                    elif saved is not None:
                        final_path = folder / 'tasks' / task_id / 'answer.txt'
                        final = final_path.read_text().strip() if final_path.is_file() else None
                        result.update(initial_answer=saved, final_answer=final,
                                      initial_correct=correct(task_id, saved, tasks[task_id]['check']),
                                      final_correct=correct(task_id, final, tasks[task_id]['check']))
                        assert result['final_correct'] == score_map[task_id]['passed']
                        output['noncode_reviews'].append(result)
                if row.get('saved') and stage != 'review':
                    prior[task_id] = candidate(row)
            aligned = len(flattened) == len(calls) and all(
                event['task_id'] == call['task_id'] and event.get('cost_units') == call.get('cost_units')
                for (_, event), call in zip(flattened, calls))
            invalid_details = [dict(task_id=event['task_id'], stage=stage, cost_units=event.get('cost_units'),
                                    finish_reason=call.get('finish_reason'), max_output=call.get('max_output'),
                                    completion_tokens=(call.get('usage') or {}).get('completion_tokens'))
                               for (stage, event), call in zip(flattened, calls)
                               if event.get('status') == 'ok' and action(event.get('text')) is None]
            assert aligned
            summary = read(folder / 'summary.json')
            assert sum(stage_cost.values()) == summary['spent_units']
            passed = {row['task_id'] for row in scores['by_task'] if row['passed']}
            passed_sets.append(passed)
            output['runs'].append(dict(policy=policy, run=run_name, reward=scores['total_reward'],
                passed=scores['count_passed'], spent_units=summary['spent_units'],
                failed=[row['task_id'] for row in scores['by_task'] if row['solvable'] and not row['passed']],
                stage_cost_units=dict(stage_cost), stage_calls=dict(stage_calls),
                invalid_cost_units=dict(invalid_cost), invalid_calls=dict(invalid_calls),
                invalid_details=invalid_details, call_event_alignment=aligned,
                persistent_failure_cost_units={task_id: task_cost[task_id]
                                               for task_id in ['next_005', 'next_006', 'next_007']},
                sensitivity_excluding_ambiguous_006_007=dict(
                    excluded=['next_006', 'next_007'],
                    reward=sum(row['reward'] for row in scores['by_task']
                               if row['passed'] and row['task_id'] not in {'next_006', 'next_007'}),
                    attainable=sum(row['reward'] for row in scores['by_task']
                                   if row['solvable'] and row['task_id'] not in {'next_006', 'next_007'}),
                    passed=sum(row['passed'] for row in scores['by_task']
                               if row['task_id'] not in {'next_006', 'next_007'}), solvable=30)))
            for name in ['score.json', 'task_results.json', 'summary.json', 'calls.jsonl']:
                path = folder / name
                output['source_hashes'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    solvable = {row['task_id'] for row in scores['by_task'] if row['solvable']}
    output['never_passed_in_eight_runs'] = sorted(solvable - set.union(*passed_sets))
    engineering = EXPERIMENTS / 'engineering/tasks'
    output['engineering100_ambiguous_overlap'] = []
    for task_id in ['next_006', 'next_007']:
        original, combined = DATA / 'tasks' / task_id, engineering / task_id
        paths = [path for path in original.rglob('*') if path.is_file() and '__pycache__' not in path.parts]
        output['engineering100_ambiguous_overlap'].append(dict(task_id=task_id,
            exact_public_copy=all((combined / path.relative_to(original)).read_bytes() == path.read_bytes()
                                  for path in paths)))
    output['sensitivity_notice'] = 'Diagnostic only; frozen scores and oracle remain unchanged; costs are not recomputed'
    destination = Path(__file__).with_suffix('.json')
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(runs=len(output['runs']), never_passed=output['never_passed_in_eight_runs'],
                          noncode_reviews=len(output['noncode_reviews']), code_reviews=len(output['code_reviews']))))


if __name__ == '__main__':
    main()
