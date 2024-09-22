# Libs >>>
import inspect
import random
from .print_color import print_colored as cprint


# Func >>>
def Make_print_sig(
    color=random.choice(["Fore.RED", "Fore.YELLOW", "Fore.GREEN", "Fore.CYAN", "Fore.MAGENTA", "Fore.BLUE"]),
):
    """
    Generates a formatted string representing the signature of the calling function.

    This function is intended to be used within other functions to provide a formatted
    print statement that identifies the calling function. It retrieves the name of the
    calling function using the `inspect` module, and returns a formatted string that
    can be printed to provide context about the current function execution.

    Returns:
        str: A formatted string representing the signature of the calling function.
    """
    # Get the caller name
    try:
        caller_frame = inspect.stack()[1]
        caller_name = caller_frame.function
    except IndexError:
        raise IndentationError("The function should not be ran in the global scope.")
    # Make + return func sig
    return cprint(
        f"\\<Func.<{color}>{caller_name}<Style.RESET_ALL>\\>",
        end="",
        return_string=True,
    )
