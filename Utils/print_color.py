# Libs >>>
import re
import colorama
from typing import Optional

# Conf >>>
print_sig = ""

# Main >>>
class ColorAttributeError(Exception):
    """Exception raised for errors in the color attributes."""

    def __init__(self, attr, message="Color attribute not found"):
        self.attr = attr
        self.message = message
        super().__init__(f"{message}: {attr}")

def set_sig(sig):
    # Loading the global print sig var and setting it to the input argument.
    global print_sig
    print_sig = sig
    # End
    
def print_colored(
    text: str, reset: bool = True, return_string: bool = False, end: str = "\n"
) -> Optional[str]:
    """
    Prints or returns the input text with color codes, interpreting escaped sequences as literal text.
    Dynamically checks and supports all attributes from the Colorama library.

    Args:
        text (str): The text to be printed or returned with color codes.
        reset (bool, optional): If True, resets the color at the end. Defaults to True.
        return_string (bool, optional): If True, returns the color-coded string instead of printing. Defaults to False.
        end (str, optional): The end character to be printed. Defaults to "\n".

    Returns:
        Optional[str]: The color-coded string if return_string is True. Otherwise, None.

    Raises:
        ColorAttributeError: If a Colorama attribute in the text is not found.

    Examples:
        >>> print_colored("This is a <Fore.RED>red text")
        This is a red text
        >>> print_colored("<t.warn>This is a warning.")
        'Warning: warning message'
    """
    # Initialize Colorama + load the global print sig var
    global print_sig

    # Defining the keywords
    keywords = {
        "warn": f"{colorama.Style.BRIGHT}{colorama.Fore.RED}Warning:{colorama.Style.RESET_ALL}{colorama.Fore.YELLOW} ",
        "err": f"{colorama.Style.BRIGHT}{colorama.Fore.RED}Error:{colorama.Style.RESET_ALL}{colorama.Fore.RED} ",
        "info": f"{colorama.Style.BRIGHT}{colorama.Fore.LIGHTBLUE_EX}Info:{colorama.Style.RESET_ALL}{colorama.Fore.WHITE} ",
        "debug": f"{colorama.Style.BRIGHT}{colorama.Fore.CYAN}Debug:{colorama.Style.RESET_ALL}{colorama.Fore.WHITE} ",
        "reset": colorama.Style.RESET_ALL,
    }

    # Handle escaped '<' and '>'
    text = text.replace(r"\<", "<").replace(r"\>", ">")

    # Find all potential Colorama attributes in the text
    matches = re.findall(r"<(\w+)\.(\w+)>", text)

    # Replace custom attributes with Colorama attributes
    for module_name, attr_name in matches:
        if module_name == "t":
            if attr_name in keywords:
                text = text.replace(f"<{module_name}.{attr_name}>", keywords[attr_name])
            else:
                raise ColorAttributeError(f"{module_name}.{attr_name}")
        else:
            # Check if the module and attribute exist in Colorama
            try:
                color_attr = getattr(getattr(colorama, module_name), attr_name)
                text = text.replace(f"<{module_name}.{attr_name}>", color_attr)
            except AttributeError:
                raise ColorAttributeError(f"{module_name}.{attr_name}")

    # If reset is True, append the reset code
    if reset:
        text += colorama.Style.RESET_ALL

    # Add the end sequence + print sig
    text = ''.join((print_sig , text, end))

    # If return_string is True, return the color-coded string
    if return_string:
        return text

    # Otherwise, print the color-coded string
    print(text, end="")


# Example usage
if __name__ == "__main__":
    print_colored("This is a <Fore.RED>red text")
    print(print_colored("<t.warn>This is a warning.", return_string=True))
