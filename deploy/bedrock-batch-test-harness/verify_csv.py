"""CSV export for verification results."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RequestRecord:
  """Complete record for a single request across all stages."""

  request_id: str
  agent_id: str
  step_id: str | None
  input_record_id: str | None
  input_status: str
  output_status: str
  outcome_status: str
  error_msg: str
  warnings: str
  tech_keywords: str


class RequestTracker:
  """Tracks requests across all validation stages."""

  def __init__(self) -> None:
    self.requests: dict[str, RequestRecord] = {}

  def add_submit_request(
    self,
    request_id: str,
    agent_id: str,
    step_id: str | None,
  ) -> None:
    """Add submit request data."""
    if request_id not in self.requests:
      self.requests[request_id] = RequestRecord(
        request_id=request_id,
        agent_id=agent_id,
        step_id=step_id,
        input_record_id=None,
        input_status="MISSING",
        output_status="MISSING",
        outcome_status="MISSING",
        error_msg="",
        warnings="",
        tech_keywords="",
      )
    else:
      record = self.requests[request_id]
      record.agent_id = agent_id
      record.step_id = step_id

  def set_input_status(
    self,
    request_id: str,
    record_id: str,
    status: str,
  ) -> None:
    """Set input stage status."""
    if request_id not in self.requests:
      self.requests[request_id] = RequestRecord(
        request_id=request_id,
        agent_id="UNKNOWN",
        step_id=None,
        input_record_id=record_id,
        input_status=status,
        output_status="MISSING",
        outcome_status="MISSING",
        error_msg="",
        warnings="",
        tech_keywords="",
      )
    else:
      record = self.requests[request_id]
      record.input_record_id = record_id
      record.input_status = status

  def set_output_status(self, request_id: str, status: str, error: str = "") -> None:
    """Set output stage status."""
    if request_id in self.requests:
      self.requests[request_id].output_status = status
      if error:
        self.requests[request_id].error_msg = error

  def set_outcome_status(
    self,
    request_id: str,
    status: str,
    tech_keywords: str = "",
    warnings: str = "",
  ) -> None:
    """Set outcome stage status."""
    if request_id in self.requests:
      self.requests[request_id].outcome_status = status
      self.requests[request_id].tech_keywords = tech_keywords
      if warnings:
        self.requests[request_id].warnings = warnings

  def get_by_agent(self, agent_id: str) -> list[RequestRecord]:
    """Get all records for a specific agent."""
    return [r for r in self.requests.values() if r.agent_id == agent_id]

  def get_all_agent_ids(self) -> set[str]:
    """Get all unique agent IDs."""
    return {r.agent_id for r in self.requests.values() if r.agent_id != "UNKNOWN"}


def export_csv_by_agent(
  tracker: RequestTracker,
  results_dir: Path,
  timestamp: str,
) -> list[Path]:
  """Export CSV files, one per agent.

  Returns list of created file paths.
  """
  created_files: list[Path] = []
  agent_ids = tracker.get_all_agent_ids()

  for agent_id in sorted(agent_ids):
    records = tracker.get_by_agent(agent_id)
    filename = f"verification_results_{agent_id}_{timestamp}.csv"
    filepath = results_dir / filename

    with open(filepath, "w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow([
        "request_id",
        "step_id",
        "input_record_id",
        "input_status",
        "output_status",
        "outcome_status",
        "error_msg",
        "warnings",
        "tech_keywords",
      ])
      for record in sorted(records, key=lambda r: r.request_id):
        writer.writerow([
          record.request_id,
          record.step_id or "",
          record.input_record_id or "",
          record.input_status,
          record.output_status,
          record.outcome_status,
          record.error_msg,
          record.warnings,
          record.tech_keywords,
        ])

    created_files.append(filepath)

  return created_files


def export_summary_csv(
  tracker: RequestTracker,
  results_dir: Path,
  timestamp: str,
) -> Path:
  """Export summary CSV with agent-level statistics.

  Returns path to created summary file.
  """
  agent_ids = tracker.get_all_agent_ids()
  agent_stats: list[dict[str, object]] = []

  for agent_id in sorted(agent_ids):
    records = tracker.get_by_agent(agent_id)
    total = len(records)
    passed = sum(
      1
      for r in records
      if r.input_status == "PASS"
      and r.output_status == "PASS"
      and r.outcome_status == "PASS"
    )
    failed = sum(
      1
      for r in records
      if "FAIL" in r.input_status
      or "FAIL" in r.output_status
      or "FAIL" in r.outcome_status
    )
    warnings = sum(1 for r in records if r.warnings)

    agent_stats.append({
      "agent_id": agent_id,
      "total_requests": total,
      "passed": passed,
      "failed": failed,
      "warnings": warnings,
      "status": "PASS" if failed == 0 else "FAIL",
    })

  filename = f"verification_results_summary_{timestamp}.csv"
  filepath = results_dir / filename

  with open(filepath, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
      "agent_id",
      "total_requests",
      "passed",
      "failed",
      "warnings",
      "status",
    ])
    for stat in agent_stats:
      writer.writerow([
        stat["agent_id"],
        stat["total_requests"],
        stat["passed"],
        stat["failed"],
        stat["warnings"],
        stat["status"],
      ])

  return filepath
