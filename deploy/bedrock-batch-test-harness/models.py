"""Result tracking for verification stage."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass
class StageResult:
  """Result of a single validation stage."""

  stage_name: str
  file_path: str
  row_count: int
  errors: list[str] = field(default_factory=list)
  warnings: list[str] = field(default_factory=list)

  @property
  def passed(self) -> bool:
    return len(self.errors) == 0

  @property
  def has_warnings(self) -> bool:
    return len(self.warnings) > 0


@dataclass
class VerificationResult:
  """Overall verification result."""

  passed: bool
  input_stage: StageResult
  submit_stage: StageResult
  output_stage: StageResult
  outcome_stage: StageResult
  errors: list[str] = field(default_factory=list)
  warnings: list[str] = field(default_factory=list)
  outcome_keywords: dict[int, object] = field(default_factory=dict)
  outcome_warnings_by_index: dict[int, list[str]] = field(default_factory=dict)

  @property
  def all_stages(self) -> list[StageResult]:
    return [
      self.input_stage,
      self.submit_stage,
      self.output_stage,
      self.outcome_stage,
    ]

  @property
  def total_rows(self) -> int:
    return sum(stage.row_count for stage in self.all_stages)

  @property
  def total_errors(self) -> int:
    return len(self.errors) + sum(len(stage.errors) for stage in self.all_stages)

  @property
  def total_warnings(self) -> int:
    return len(self.warnings) + sum(len(stage.warnings) for stage in self.all_stages)
