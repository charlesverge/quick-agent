from __future__ import annotations

from quick_agent.executor import AgentExecutor
from quick_agent.models.batch_request import (
    BatchImportRequest,
    BatchSubmitRequest,
)

import logging

logger = logging.getLogger(__name__)


class AgentProcessor:
    def __init__(self, executor: AgentExecutor) -> None:
        self._executor = executor

    async def run_batch(self, batch_request: BatchSubmitRequest) -> BatchImportRequest:
        """Execute batch request directly against LLM API with full tool-call loop."""
        request = batch_request
        while True:
            batch_import = await self._executor._local_batch_call(request)
            outcome = self._executor.import_outcome(batch_import=batch_import)

            if outcome.tool_calls is not None:
                if request.tool_call_rounds() >= request.max_tool_calls:
                    raise ValueError(
                        f"Max tool call rounds reached for request_id={request.request_id}: "
                        f"max_tool_calls={request.max_tool_calls}"
                    )
                pending = outcome.pending_submit_request
                if pending is None:
                    raise ValueError(
                        "tool_use outcome is missing pending_submit_request."
                    )
                executed = await self._executor._execute_tool_calls(outcome.tool_calls)
                request = self._executor._build_next_request_with_tool_results(
                    tool_calls=outcome.tool_calls,
                    executed=executed,
                    submit_request=pending,
                )
                continue

            if outcome.next_request is not None:
                request = outcome.next_request
                continue

            return batch_import
