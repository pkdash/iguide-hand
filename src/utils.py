#!/usr/bin/env python3
"""
Utility functions for HAND Dam Workflow

Common functions shared across multiple workflow scripts.
"""

import os
import sys
import shutil
import shlex
import subprocess
import logging

logger = logging.getLogger(__name__)


def setup_taudem_path(taudem_path):
    """
    Add TauDEM to system PATH.
    
    Args:
        taudem_path: Path to TauDEM installation
    """
    current_path = os.environ.get("PATH", "")
    if taudem_path not in current_path:
        os.environ["PATH"] = f"{taudem_path}:{current_path}"
        logger.info(f"Added TauDEM path: {taudem_path}")

    # Check if TauDEM is available
    mpiexec_path = shutil.which("mpiexec")
    if mpiexec_path is None:
        logger.warning("mpiexec not found in PATH. TauDEM commands may fail.")
    else:
        logger.info(f"Found mpiexec at: {mpiexec_path}")


def run_taudem_command(command, description="TauDEM command", cwd=None):
    """
    Execute a TauDEM command with error handling.

    Args:
        command: Command string to execute
        description: Description for logging
        cwd: Working directory for command execution
    """
    logger.info(f"Running {description}: {command}")
    cmd_list = shlex.split(command)
    try:
        result = subprocess.run(
            cmd_list,
            shell=False,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Warnings: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"TauDEM command failed: {' '.join(cmd_list)}")
        logger.error(f"Error: {e.stderr}")
        sys.exit(1)
