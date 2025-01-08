import logging
import os
import shutil

def backup_file(file_path):
    """Backup the original file before upgrading."""
    backup_path = file_path + ".bak"
    shutil.copy(file_path, backup_path)
    logging.info(f"Backup created: {backup_path}")
    return backup_path

def rollback_file(file_path):
    """Rollback to the backup file in case of failure."""
    backup_path = file_path + ".bak"
    if os.path.exists(backup_path):
        shutil.copy(backup_path, file_path)
        logging.info(f"Rollback successful: {file_path} restored from {backup_path}")
        return True
    logging.warning(f"No backup found for rollback: {file_path}")
    return False
