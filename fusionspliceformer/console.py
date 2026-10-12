"""Compact terminal output for input preparation and completed ensemble events."""
import sys
import threading
import time


class Activity:
    """A single preparation line, including an elapsed-time heartbeat on terminals."""
    def __init__(self, label, enabled=True):
        self.label, self.enabled = label, enabled
        self.stop = threading.Event()
        self.thread = None
        self.width = 0
        self.final_text = None

    def write(self, text, final=False):
        if sys.stdout.isatty():
            sys.stdout.write('\r'+text.ljust(self.width)+('\n' if final else ''))
            sys.stdout.flush()
            self.width = max(self.width, len(text))
        elif final:
            print(text, flush=True)

    def __enter__(self):
        if self.enabled:
            self.started = time.perf_counter()
            if sys.stdout.isatty():
                self.write(self.label+'...')
                self.thread = threading.Thread(target=self.heartbeat, daemon=True)
                self.thread.start()
            else:
                print(self.label+'...', flush=True)
        return self

    def heartbeat(self):
        while not self.stop.wait(1):
            self.write(f'{self.label}... elapsed {time.perf_counter()-self.started:.0f}s')

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        if self.thread:
            self.thread.join()
        if self.enabled:
            text = self.final_text if kind is None and self.final_text else self.label+('... done' if kind is None else '... interrupted')
            self.write(text, final=True)


class EventProgress:
    """Counts only events whose complete ensemble scores have been computed."""
    def __init__(self, metadata, initial, formatter, interval=0.5, small_limit=10):
        self.metadata = metadata
        self.keys = metadata.index.tolist()
        self.total = len(self.keys)
        self.scores = dict(initial)
        self.initial_keys = set(initial)
        self.formatter = formatter
        self.small = self.total <= small_limit
        self.started = time.perf_counter()
        self.last = self.started
        self.interval = interval if sys.stdout.isatty() else 5.0
        self.cursor = self.width = 0
        self.line_open = False
        if initial:
            print(f'Resuming: {len(initial):,}/{self.total:,} events already completed.', flush=True)
        if not self.small:
            self.render(force=True)

    def advance(self, keys, scores):
        for key, score in zip(keys, scores):
            self.scores[key] = float(score)
        if self.small:
            # Keep short per-event output in original input order, even with
            # interleaved acceptor/donor batches.
            while self.cursor < self.total and self.keys[self.cursor] in self.scores:
                key = self.keys[self.cursor]
                self.cursor += 1
                if key in self.initial_keys:
                    continue
                row = self.metadata.loc[key]
                chrom, pos, ref, alt = row.variant_key.split(':')
                score = self.scores[key]
                call = 'Pathogenic' if score >= 0.5 else 'Benign'
                site = row.physical_site_id.split(':', 1)[1]
                print(f'{self.cursor}/{self.total} | {chrom}:{pos} {ref}>{alt} | '
                      f'target={site} | score={self.formatter(score)} | {call}', flush=True)
        else:
            self.render()

    def render(self, force=False):
        now = time.perf_counter()
        completed = len(self.scores)
        if not force and completed != self.total and now-self.last < self.interval:
            return
        elapsed = now-self.started
        new = completed-len(self.initial_keys)
        eta = f'{elapsed*(self.total-completed)/new:.1f}s' if new else '--'
        if completed == self.total:
            eta = '0.0s'
        text = (f'Predicting: {completed:,}/{self.total:,} ({100*completed/self.total:.1f}%) | '
                f'elapsed {elapsed:.1f}s | ETA {eta}')
        if sys.stdout.isatty():
            sys.stdout.write('\r'+text.ljust(self.width))
            sys.stdout.flush()
            self.width = max(self.width, len(text))
            self.line_open = True
        else:
            print(text, flush=True)
        self.last = now

    def close(self):
        if self.line_open:
            print(flush=True)
            self.line_open = False


class ChunkProgress:
    """One live line spanning feature preparation and prediction of all chunks."""
    def __init__(self, total, initial=0):
        self.total, self.initial = total, initial
        self.state = (initial, 'preparing')
        self.stop = threading.Event()
        self.width = 0

    def __enter__(self):
        self.started = time.perf_counter()
        self.render()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()
        return self

    def update(self, completed, stage):
        self.state = (completed, stage)

    def heartbeat(self):
        while not self.stop.wait(1 if sys.stdout.isatty() else 5):
            self.render()

    def render(self, final=False):
        completed, stage = self.state
        elapsed = time.perf_counter()-self.started
        advanced = completed-self.initial
        eta = f'{elapsed*(self.total-completed)/advanced:.0f}s' if advanced > 0 else '--'
        if completed == self.total: eta = '0s'
        percent = 100*completed/self.total if self.total else 100
        text = (f'Processing: {completed:,}/{self.total:,} ({percent:.1f}%) | {stage} | '
                f'elapsed {elapsed:.1f}s | ETA {eta}')
        if sys.stdout.isatty():
            sys.stdout.write('\r'+text.ljust(self.width)+('\n' if final else ''))
            sys.stdout.flush()
            self.width = max(self.width, len(text))
        else:
            print(text, flush=True)

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join()
        if kind is not None:
            self.update(self.state[0], 'interrupted')
        self.render(final=True)
