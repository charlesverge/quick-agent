#!/usr/bin/env python3
"""Primary entrypoint for the bedrock batch test harness."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import execution as execution_stage
import setup as setup_stage
import setup_code_rule as setup_code_rule_stage
import verify as verify_stage
import verify_code_rule as verify_code_rule_stage
from settings import HarnessSettings
from settings import load_runtime_settings
from settings import load_settings


def _configure_logger(settings: HarnessSettings) -> logging.Logger:
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    log_name = f"run-{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
    log_path = settings.logs_dir / log_name
    logger = logging.getLogger("bedrock_batch_test_harness")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def _cleanup(settings: HarnessSettings, logger: logging.Logger) -> None:
    if settings.runtime_dir.exists():
        shutil.rmtree(settings.runtime_dir)
        logger.info(f"cleanup: removed {settings.runtime_dir}")
    if settings.logs_dir.exists():
        shutil.rmtree(settings.logs_dir)
        logger.info(f"cleanup: removed {settings.logs_dir}")
    if settings.input_jsonl.exists():
        settings.input_jsonl.unlink()
        logger.info(f"cleanup: removed {settings.input_jsonl}")
    if settings.output_jsonl.exists():
        settings.output_jsonl.unlink()
        logger.info(f"cleanup: removed {settings.output_jsonl}")
    if settings.outcomes_jsonl.exists():
        settings.outcomes_jsonl.unlink()
        logger.info(f"cleanup: removed {settings.outcomes_jsonl}")


def _selected_stage_flags(args: argparse.Namespace) -> tuple[bool, bool, bool]:
    has_selection = args.setup or args.execute or args.verify
    if not has_selection:
        return True, True, True
    return args.setup, args.execute, args.verify


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="default")
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--no-tear-down", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    settings = load_settings(fixture_name=args.fixture)
    logger = _configure_logger(settings)

    if args.cleanup:
        _cleanup(settings, logger)
        return 0

    run_setup, run_execution, run_verify = _selected_stage_flags(args)
    active_settings = settings
    try:
        if run_setup:
            n_code_rules = len(setup_code_rule_stage._list_pairs(active_settings.harness_root))
            logger.info("stage: setup > start")
            active_settings = setup_stage.run(active_settings, reserved=n_code_rules)
            logger.info("stage: setup > complete")
            logger.info("stage: setup code-rules > start")
            setup_code_rule_stage.run(active_settings)
            logger.info("stage: setup code-rules > complete")
        elif run_execution or run_verify:
            active_settings = load_runtime_settings(harness_root=settings.harness_root)

        if run_execution:
            logger.info("stage: execution > start")
            execution_stage.run(active_settings)
            logger.info("stage: execution > complete")

        if run_verify:
            logger.info("stage: verify > start")
            verify_stage.run(active_settings)
            logger.info("stage: verify > complete")
            logger.info("stage: verify code-rules > start")
            verify_code_rule_stage.run(active_settings)
            logger.info("stage: verify code-rules > complete")

        if args.no_tear_down:
            logger.info("lifecycle: no-tear-down enabled")
        logger.info("harness: success")
        return 0
    except Exception as error:
        logger.error(f"harness: failure > {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
