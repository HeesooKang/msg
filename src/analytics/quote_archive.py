"""Append-only quote evidence, never a model or an admission dependency."""

import gzip
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

from src.models import Quote

logger = logging.getLogger("kis_trader.quote_archive")


class QuoteArchive:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._date = ""
        self._failed = False

    def record(self, quotes: Sequence[Quote], *, now: datetime, universe: Sequence[str]) -> None:
        day = now.date().isoformat()
        payload = dict(observed_timestamp=now.isoformat(), reset_history=self._date != day,
                       universe=list(universe), quotes=[
                           dict(asdict(q), timestamp=q.timestamp.isoformat()) for q in quotes])
        path = self.root / day[:4] / day[5:7] / f"quote-batches.{day}.jsonl.gz"
        if self._append(path, payload):
            self._date = day

    def record_discovery(self, *, now: datetime, environment: str, params: dict,
                         rows: Sequence[dict], symbols: Sequence[str]) -> None:
        day = now.date().isoformat()
        fields = ("stck_shrn_iscd", "data_rank", "hts_kor_isnm", "prdy_ctrt",
                  "stck_prpr", "lwpr_vrss_prpr_rate")
        payload = dict(observed_timestamp=now.isoformat(), environment=environment,
                       params=dict(params), symbols=list(symbols),
                       rows=[{key: row[key] for key in fields if key in row} for row in rows])
        path = self.root / day[:4] / day[5:7] / f"ranking-snapshots.{day}.jsonl.gz"
        self._append(path, payload)

    def _append(self, path: Path, payload: dict) -> bool:
        try:
            data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
            compressed = gzip.compress(data, compresslevel=1, mtime=0)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as handle:
                handle.write(compressed)
            self._failed = False
            return True
        except OSError as exc:
            if not self._failed:
                logger.warning("시세 원장 기록 실패 [%s]: %s", path.name, exc)
            self._failed = True
            return False
