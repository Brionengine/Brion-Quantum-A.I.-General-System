from asyncio import subprocess
import logging


def run_tests(test_dir="./tests"):
    """Run all Python tests in the specified directory."""
    try:
        result = subprocess.run(["pytest", test_dir], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info("All tests passed successfully!")
            return True
        else:
            logging.error(f"Tests failed: {result.stderr}")
            return False
    except Exception as e:
        logging.error(f"Failed to run tests: {e}")
        return False
def upgrade_and_test(self, file_name):
    """Upgrade a file and run tests to validate changes."""
    backup = backup_file(file_name) # type: ignore
    upgraded_file = self.upgrade_code(file_name)
    if upgraded_file and run_tests():
        logging.info(f"Upgrade and tests successful for {file_name}.")
        return upgraded_file
    else:
        rollback_file(file_name) # type: ignore
        logging.error(f"Upgrade failed for {file_name}. Rolled back to original.")
        return None
