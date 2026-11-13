"""Bounded raw SSE accounting; bytes pass unchanged to the native SDK."""
import json
import time
import httpx
from submission.budget import normalize_actual_usage


class AccountedStream(httpx.AsyncByteStream):
    def __init__(self, stream, state, budget):
        self.stream, self.state, self.budget = stream, state, budget
        self.buffer = b''
        self.size = 0
        self.content = []
        self.usage = None
        self.finishes = []
        self.finished = False
        state.update(stream_done=False, stream_complete=False, raw_usage_count=0)

    def feed(self, chunk):
        self.size += len(chunk)
        if self.size > 2_000_000:
            raise ValueError('SSE byte limit exceeded')
        self.buffer += chunk
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            line = line.rstrip(b'\r')
            if self.state['stream_done'] and line:
                raise ValueError('SSE bytes after DONE')
            if not hasattr(self, 'event'):
                self.event = []
            if line == b'':
                if self.event:
                    self.event_data(b'\n'.join(self.event).decode('utf-8'))
                self.event = []
            elif line.startswith(b'data:'):
                self.event.append(line[5:].lstrip(b' '))

    def event_data(self, text):
        if self.state['stream_done']:
            raise ValueError('SSE data after DONE')
        if text == '[DONE]':
            self.state['stream_done'] = True
            valid = (self.state['raw_usage_count'] == 1 and len(self.finishes) == 1
                     and self.finishes[0] in ('stop', 'length')
                     and normalize_actual_usage({'usage': self.usage}) is not None)
            if not valid:
                raise ValueError('SSE missing or invalid final usage/finish')
            # Match E132's exported metadata whitelist; billing and raw audit
            # continue to consume the unmodified provider usage object.
            exported_usage = {key: value if type(value) is int and value >= 0 else None
                              for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
                              for value in [self.usage.get(key)]}
            details = self.usage.get('completion_tokens_details')
            if (isinstance(details, dict) and type(details.get('reasoning_tokens')) is int
                    and details['reasoning_tokens'] >= 0):
                exported_usage['completion_tokens_details'] = {'reasoning_tokens': details['reasoning_tokens']}
            self.state.update(usage=exported_usage, raw_usage_numeric=self.usage,
                finish_reason=self.finishes[0], content=''.join(self.content),
                nonempty_final_content=bool(''.join(self.content).strip()))
            return
        item = json.loads(text)
        if not isinstance(item, dict):
            raise ValueError('SSE event is not an object')
        choices = item.get('choices')
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError('SSE invalid choices')
        usage = item.get('usage')
        # The captured proxy final usage trailer has one no-op choice, with
        # finish_reason omitted (SDK-equivalent to null). Never accept content.
        if (len(self.finishes) == 1 and self.usage is None and isinstance(usage, dict)
                and len(choices) == 1 and isinstance(choices[0], dict)
                and set(choices[0]) <= {'index', 'delta', 'finish_reason'}
                and type(choices[0].get('index')) is int and choices[0]['index'] == 0
                and choices[0].get('delta') == {} and choices[0].get('finish_reason') is None):
            choices = []
        if choices:
            if self.finishes or self.usage is not None:
                raise ValueError('SSE choice after final finish or usage')
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get('index', 0) != 0:
                raise ValueError('SSE invalid choice')
            delta = choice.get('delta')
            if not isinstance(delta, dict) or delta.get('tool_calls') or delta.get('function_call'):
                raise ValueError('SSE unexpected tool or delta')
            content = delta.get('content')
            if content is not None:
                if not isinstance(content, str):
                    raise ValueError('SSE nontext content')
                self.content.append(content)
            if choice.get('finish_reason') is not None:
                self.finishes.append(choice['finish_reason'])
        usage = item.get('usage')
        if usage is not None:
            self.state['raw_usage_count'] += 1
            # Only one final snapshot is unambiguous. Interim/repeated usage is
            # rejected rather than summed or inferred to be cumulative.
            if (self.state['raw_usage_count'] != 1 or not isinstance(usage, dict)
                    or len(self.finishes) != 1
                    or (choices and choices[0].get('finish_reason') is None)):
                raise ValueError('SSE interim or duplicate usage')
            self.usage = usage

    async def __aiter__(self):
        if self.finished:
            return
        iterator = self.stream.__aiter__()
        async for chunk in iterator:
            if chunk and 'first_byte_seconds' not in self.state:
                self.state['first_byte_seconds'] = time.monotonic() - self.state['stream_started']
            self.feed(chunk)
            if self.state['stream_done']:
                # The SDK stops reading at DONE. Validate transport EOF first;
                # an error/trailing event must not settle an unknown ticket.
                async for trailing in iterator:
                    self.feed(trailing)
                if self.buffer.strip() or getattr(self, 'event', []):
                    raise ValueError('SSE trailing incomplete event')
                self.state['cost_units'] = self.budget.settle(self.state['ticket'], {'usage': self.usage})
                self.state['settled'] = True
                self.state['stream_complete'] = True
                self.finished = True
                yield chunk
                return
            yield chunk

    async def aclose(self):
        await self.stream.aclose()
