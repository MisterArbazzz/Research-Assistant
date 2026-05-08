"""Async pub/sub bus for adapter-side events.

LangGraph emits its own state-update stream via `graph.astream(...)`. Our
mock adapters need to surface their side-effects (a Slack message was
'sent', a ClickUp task was 'created') in the same SSE stream the UI is
already consuming. This bus is the fan-in.

Each run subscribes once via `subscribe(run_id)` and gets back an asyncio
Queue. Adapters publish via `publish(run_id, event_type, payload)`. The
/runs/{id}/events endpoint drains the queue alongside graph state updates
and forwards them as SSE events.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class SSEBus:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[dict[str, Any]]] = defaultdict(asyncio.Queue)

    def subscribe(self, run_id: str) -> asyncio.Queue[dict[str, Any]]:
        return self._queues[run_id]

    def unsubscribe(self, run_id: str) -> None:
        self._queues.pop(run_id, None)

    async def publish(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        queue = self._queues.get(run_id)
        if queue is None:
            logger.warning("publish to unsubscribed run_id", extra={"run_id": run_id})
            return
        await queue.put({"event": event_type, "data": json.dumps(payload, default=str)})


sse_bus = SSEBus()
