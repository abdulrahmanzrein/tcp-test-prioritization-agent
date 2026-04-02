"""
Agent configuration — controls pilot vs production mode.

Pilot mode:   Reads all features from the pre-extracted CSV dataset.
Production mode: Extracts features from real sources (SciTools Understand,
                 git history, pydriller, coverage analysis).
"""

from enum import Enum


class AgentMode(Enum):
    PILOT = "pilot"            # Read from CSV dataset (default)
    PRODUCTION = "production"  # Extract from real sources


# ── Global config — set once at startup ──────────────────────────────
_config = {
    "mode": AgentMode.PILOT,
    "dataset_path": None,       # Used in pilot mode — path to the CSV
    # Production mode settings (only needed when mode == PRODUCTION)
    "project_path": None,       # Path to the project's git repo
    "test_path": None,          # Relative path to test source code root
    "ci_data_path": None,       # Path to CI datasource (RTP-Torrent / Travis)
    "output_path": None,        # Where to save intermediate outputs
    "language": "java",         # Project language
    "level": "file",            # Analysis level: "file" or "function"
    "build_window": 6,          # Number of recent builds for windowed features
}


def set_mode(mode: AgentMode, **kwargs):
    """Set the agent mode and any additional config overrides."""
    _config["mode"] = mode
    _config.update(kwargs)


def get_mode() -> AgentMode:
    return _config["mode"]


def get_config() -> dict:
    return _config.copy()
