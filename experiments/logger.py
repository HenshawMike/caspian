"""Structured experiment logging system for Project Caspian.

Logs environment states, trajectories, agent metrics, and verification results
in standardized JSON format for scientific traceability and automated report generation.
"""

import datetime
import hashlib
import json
import os
from typing import Dict, Any, List, Optional


class ExperimentLogger:
    """Standardized logger for Caspian experiments."""

    def __init__(
        self,
        experiment_id: str,
        phase: str = "Phase 0 — Research & Safety Foundation",
        description: str = "",
        log_dir: str = "logs",
    ):
        self.experiment_id = experiment_id
        self.phase = phase
        self.description = description
        self.log_dir = log_dir
        self.created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        self.config: Dict[str, Any] = {}
        self.commands_executed: List[str] = []
        self.test_results: Dict[str, Any] = {}
        self.trajectories: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.semantic_audit: Dict[str, Any] = {}
        self.observations_and_notes: List[str] = []

    def set_config(self, config: Dict[str, Any]) -> None:
        """Record environment and agent configuration."""
        self.config = config

    def log_command(self, cmd: str) -> None:
        """Record a CLI or test command executed during the experiment."""
        self.commands_executed.append(cmd)

    def log_test_results(self, total: int, passed: int, failed: int, details: List[str]) -> None:
        """Record unit and acceptance test outcomes."""
        self.test_results = {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "success_rate": round(passed / total * 100.0, 2) if total > 0 else 0.0,
            "details": details,
        }

    def log_trajectory(self, run_name: str, seed: int, steps: List[Dict[str, Any]]) -> str:
        """Record a step-by-step trajectory and return its SHA-256 hash."""
        traj_str = json.dumps(steps, sort_keys=True)
        traj_hash = hashlib.sha256(traj_str.encode("utf-8")).hexdigest()
        self.trajectories[run_name] = {
            "seed": seed,
            "total_steps": len(steps),
            "sha256_hash": traj_hash,
            "steps": steps,
        }
        return traj_hash

    def log_metrics(self, metric_group: str, data: Dict[str, Any]) -> None:
        """Record aggregated benchmark and evaluation metrics."""
        self.metrics[metric_group] = data

    def log_semantic_audit(self, passed: bool, tokens_scanned: List[str], leakages_found: int, details: str) -> None:
        """Record semantic firewall audit outcome."""
        self.semantic_audit = {
            "passed": passed,
            "tokens_scanned": tokens_scanned,
            "leakages_found": leakages_found,
            "details": details,
        }

    def add_note(self, note: str) -> None:
        """Add qualitative observation or scientific note."""
        self.observations_and_notes.append(note)

    def save_json(self) -> str:
        """Persist all experiment records to a JSON file."""
        os.makedirs(self.log_dir, exist_ok=True)
        filepath = os.path.join(self.log_dir, f"{self.experiment_id}.json")
        data = {
            "experiment_id": self.experiment_id,
            "phase": self.phase,
            "created_at": self.created_at,
            "description": self.description,
            "config": self.config,
            "commands_executed": self.commands_executed,
            "test_results": self.test_results,
            "trajectories": self.trajectories,
            "metrics": self.metrics,
            "semantic_audit": self.semantic_audit,
            "notes": self.observations_and_notes,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return filepath
