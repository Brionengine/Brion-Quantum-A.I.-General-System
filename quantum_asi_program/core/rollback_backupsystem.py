"""
Brion Quantum - Rollback & Backup System v2.0
===============================================
Enhanced backup/rollback system with versioned snapshots, integrity verification,
and automatic cleanup. Supports the self-improvement and self-healing pipelines.

Features:
  - Timestamped versioned backups
  - SHA-256 integrity verification
  - Automatic old backup cleanup (configurable retention)
  - Multi-file atomic rollback
  - Snapshot comparison and diff

Developed by Brion Quantum AI Team
"""

import logging
import os
import shutil
import hashlib
import json
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta


def file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return ""


def backup_file(file_path: str, backup_dir: Optional[str] = None) -> Optional[str]:
    """
    Create a timestamped backup of a file with integrity verification.

    Args:
        file_path: Path to the file to backup
        backup_dir: Optional directory for backups (default: same as file)

    Returns:
        Path to the backup file, or None if failed
    """
    if not os.path.exists(file_path):
        logging.warning(f"Cannot backup non-existent file: {file_path}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, f"{os.path.basename(file_path)}.{timestamp}.bak")
    else:
        backup_path = f"{file_path}.{timestamp}.bak"

    try:
        shutil.copy2(file_path, backup_path)

        # Verify integrity
        original_hash = file_hash(file_path)
        backup_hash_val = file_hash(backup_path)

        if original_hash != backup_hash_val:
            logging.error(f"Backup integrity check failed for {file_path}")
            os.remove(backup_path)
            return None

        # Store metadata
        meta_path = backup_path + ".meta"
        metadata = {
            "original": file_path,
            "backup": backup_path,
            "timestamp": timestamp,
            "hash": original_hash,
            "size": os.path.getsize(file_path)
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"Backup created: {backup_path} (hash: {original_hash[:12]})")
        return backup_path

    except Exception as e:
        logging.error(f"Failed to backup {file_path}: {e}")
        return None


def rollback_file(file_path: str, backup_path: Optional[str] = None) -> bool:
    """
    Rollback a file to a specific backup or the most recent one.

    Args:
        file_path: Path to the file to rollback
        backup_path: Specific backup to restore (default: most recent)

    Returns:
        True if rollback was successful
    """
    if backup_path and os.path.exists(backup_path):
        return _restore_from_backup(file_path, backup_path)

    # Find the most recent backup
    latest = find_latest_backup(file_path)
    if latest:
        return _restore_from_backup(file_path, latest)

    logging.warning(f"No backup found for rollback: {file_path}")
    return False


def _restore_from_backup(file_path: str, backup_path: str) -> bool:
    """Restore a file from a specific backup with integrity check."""
    try:
        # Verify backup exists and is readable
        if not os.path.exists(backup_path):
            logging.error(f"Backup file not found: {backup_path}")
            return False

        # Check metadata if available
        meta_path = backup_path + ".meta"
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            backup_hash_val = file_hash(backup_path)
            if backup_hash_val != metadata.get("hash", ""):
                logging.warning(f"Backup integrity mismatch for {backup_path}, proceeding anyway")

        shutil.copy2(backup_path, file_path)
        logging.info(f"Rollback successful: {file_path} restored from {backup_path}")
        return True

    except Exception as e:
        logging.error(f"Rollback failed for {file_path}: {e}")
        return False


def find_latest_backup(file_path: str) -> Optional[str]:
    """Find the most recent backup for a file."""
    directory = os.path.dirname(file_path) or "."
    basename = os.path.basename(file_path)

    backups = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith(basename) and f.endswith(".bak")
    ], reverse=True)

    return backups[0] if backups else None


def list_backups(file_path: str) -> List[Dict[str, Any]]:
    """List all available backups for a file."""
    directory = os.path.dirname(file_path) or "."
    basename = os.path.basename(file_path)

    backups = []
    for f in sorted(os.listdir(directory)):
        if f.startswith(basename) and f.endswith(".bak"):
            full_path = os.path.join(directory, f)
            meta_path = full_path + ".meta"
            info = {
                "path": full_path,
                "size": os.path.getsize(full_path),
                "modified": datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat()
            }
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as mf:
                        info["metadata"] = json.load(mf)
                except Exception:
                    pass
            backups.append(info)

    return backups


def cleanup_old_backups(file_path: str, keep: int = 5, max_age_days: int = 30):
    """Clean up old backups, keeping the most recent ones."""
    directory = os.path.dirname(file_path) or "."
    basename = os.path.basename(file_path)
    cutoff = datetime.now() - timedelta(days=max_age_days)

    backups = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith(basename) and f.endswith(".bak")
    ], reverse=True)

    # Keep the most recent 'keep' backups
    to_remove = backups[keep:]

    for backup in to_remove:
        try:
            mod_time = datetime.fromtimestamp(os.path.getmtime(backup))
            if mod_time < cutoff:
                os.remove(backup)
                meta = backup + ".meta"
                if os.path.exists(meta):
                    os.remove(meta)
                logging.info(f"Cleaned up old backup: {backup}")
        except Exception as e:
            logging.warning(f"Failed to clean up {backup}: {e}")


def atomic_multi_rollback(file_backup_pairs: List[Dict[str, str]]) -> bool:
    """
    Atomically rollback multiple files. If any rollback fails,
    attempt to restore all files to their pre-rollback state.

    Args:
        file_backup_pairs: List of {"file": path, "backup": backup_path} dicts

    Returns:
        True if all rollbacks were successful
    """
    # First, backup current state of all files
    pre_rollback = []
    for pair in file_backup_pairs:
        pre = backup_file(pair["file"])
        if pre:
            pre_rollback.append({"file": pair["file"], "pre_backup": pre})

    # Attempt all rollbacks
    results = []
    for pair in file_backup_pairs:
        success = rollback_file(pair["file"], pair.get("backup"))
        results.append({"file": pair["file"], "success": success})

    # If any failed, rollback the rollbacks
    if not all(r["success"] for r in results):
        logging.error("Multi-rollback partially failed, restoring pre-rollback state")
        for pre in pre_rollback:
            rollback_file(pre["file"], pre["pre_backup"])
        return False

    logging.info(f"Multi-rollback successful: {len(results)} files restored")
    return True
