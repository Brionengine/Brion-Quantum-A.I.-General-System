# utils/helpers.py

def validate_input(data):
    """
    Validates user input.

    Args:
        data (str): Input data to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    # Implement validation logic
    return isinstance(data, str) and len(data.strip()) > 0
