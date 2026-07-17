"""The search ledger — an honest denominator for the data-mining correction.

Every correction for data-mining bias (`sidak_correction`, `deflated_sharpe`,
`whites_reality_check` / `spa_test`, and the REAL gate) needs one number: how
many variants did you actually try? That number is only honest if it counts the
whole search — including every variant you discarded — not just the winner you
kept. Report a smaller N and the correction gets weaker, the p-value gets
prettier, and the gate you built to catch data mining waves the data mining
through.

`SearchSpaceLog` is the record that makes the correction defensible: an
append-only ledger of every variant attempted, written as you search rather
than recalled afterwards. Hand its count to the correction instead of a
hardcoded integer, and the denominator becomes a measurement rather than a
self-attestation.

    >>> log = SearchSpaceLog(scope="ES:ma_cross_grid")
    >>> for fast, slow in [(10, 50), (20, 50), (10, 100), (20, 100)]:
    ...     log.record({"fast": fast, "slow": slow}, status="tried")
    >>> log.mark_selected({"fast": 20, "slow": 100}, score=0.31)
    >>> log.session_n_variants          # 5 -> feeds REAL's Sidak correction
    5

The ledger is in-memory by default. Pass `path=` to persist as JSONL, and
entries survive across runs so a later correction can re-load the full history.

Aronson's data-mining-bias correction is the underlying argument: a correction
is only valid if it sees the full set of variants attempted. An incomplete log
silently degrades an audited gate back into the point-threshold gating it exists
to prevent.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class SearchSpaceLog:
    """Append-only log of every variant tried during a search.

    Each entry is one variant: a params dict, an optional score, and a status
    ('tried', 'discarded', 'selected'). Record variants as you try them —
    including the failures — so the count reflects the search you actually ran.

    Two counts, and the difference matters (see :attr:`n_variants` and
    :attr:`session_n_variants` for which is honest when).
    """

    VALID_STATUSES = ('tried', 'discarded', 'selected')

    def __init__(self, scope: str, path: Optional[str] = None):
        """
        Args:
            scope: What dimension this log tracks — any stable label for one
                   search, e.g. 'ES:ma_cross_grid' for a parameter sweep on one
                   market, or 'universe' for a search across markets. Only
                   entries matching `scope` are re-loaded from `path`.
            path:  Optional JSONL file to persist to. Omit (the default) and the
                   ledger stays purely in memory — no side effects. If the file
                   already exists, existing entries under this `scope` are loaded
                   so counts can accumulate across runs.
        """
        self.scope = scope
        self.path = path
        self.entries: List[Dict[str, Any]] = []
        # Entries recorded via record() in THIS process, kept apart from any
        # loaded from disk — see session_n_variants.
        self._session_entries: List[Dict[str, Any]] = []
        if path and os.path.exists(path):
            self._load()

    def record(self, params: Dict[str, Any], score: Optional[float] = None,
               status: str = 'tried', **extra) -> Dict[str, Any]:
        """Record one variant. Returns the entry as stored."""
        if status not in self.VALID_STATUSES:
            raise ValueError(f"status must be one of {self.VALID_STATUSES}, got '{status}'")
        entry = {
            'scope': self.scope,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'params': params,
            'score': score,
            'status': status,
        }
        if extra:
            entry.update(extra)
        self.entries.append(entry)
        self._session_entries.append(entry)
        if self.path:
            self._append_to_file(entry)
        return entry

    def mark_selected(self, params: Dict[str, Any], score: Optional[float] = None, **extra):
        """Convenience: record the variant that was ultimately chosen."""
        return self.record(params, score=score, status='selected', **extra)

    @property
    def n_variants(self) -> int:
        """Total variants on record, **including any loaded from prior runs**.

        This is the honest denominator when your search genuinely spanned
        sessions — you explored new variants across several runs and kept the
        best of all of them. That accumulated total is what you actually mined.
        """
        return len(self.entries)

    @property
    def session_n_variants(self) -> int:
        """Variants recorded in **this process only** (excludes entries loaded
        from disk).

        This is the honest denominator when a run is idempotent — re-running the
        same search shouldn't compound its own penalty just because the ledger
        remembers the previous identical run.

        Choosing between this and :attr:`n_variants` is a real judgement, not a
        default: session-scoping protects *re-runs of the same search*, not
        *genuine new exploration*. If each run tries something new, the honest
        number drifts toward `n_variants`.
        """
        return len(self._session_entries)

    def scores(self) -> List[float]:
        """All recorded scores (skipping entries logged without one)."""
        return [e['score'] for e in self.entries if e.get('score') is not None]

    def best(self) -> Optional[Dict[str, Any]]:
        """Entry with the highest score, or None if no scores were logged."""
        scored = [e for e in self.entries if e.get('score') is not None]
        if not scored:
            return None
        return max(scored, key=lambda e: e['score'])

    def _append_to_file(self, entry: Dict[str, Any]):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    def _load(self):
        with open(self.path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get('scope') == self.scope:
                    self.entries.append(entry)

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"SearchSpaceLog(scope='{self.scope}', n_variants={self.n_variants})"
