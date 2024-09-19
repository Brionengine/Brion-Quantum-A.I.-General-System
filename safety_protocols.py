def safe_execute(function):
    """
    Decorator to safely execute functions with exception handling.
    """
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            # Log the exception
            logging.error(f"Error in {function.__name__}: {e}")
            # Return a safe response or re-raise the exception
            raise
    return wrapper
