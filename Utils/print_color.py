# Libs >>>
import re
import colorama
from typing import Optional


# Main >>>
class ColorAttributeError(Exception):
    """Exception raised for errors in the color attributes."""

    def __init__(self, attr, message="Color attribute not found"):
        self.attr = attr
        self.message = message
        super().__init__(f"{message}: {attr}")


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
    """
    # Handle escaped '<' and '>'
    text = text.replace(r"\<", "<").replace(r"\>", ">")

    # Find all potential Colorama attributes in the text
    matches = re.findall(r"<(\w+)\.(\w+)>", text)

    # Replace custom attributes with Colorama attributes
    for module_name, attr_name in matches:
        # Check if the module and attribute exist in Colorama
        if hasattr(colorama, module_name) and hasattr(
            getattr(colorama, module_name), attr_name
        ):
            color_attr = getattr(getattr(colorama, module_name), attr_name)
            text = text.replace(f"<{module_name}.{attr_name}>", color_attr)
        else:
            raise ColorAttributeError(f"{module_name}.{attr_name}")

    # Check if reset is True, append the reset code
    if reset:
        text += colorama.Style.RESET_ALL

    # Add the end seq
    text += end

    # If return_string is True, return the color-coded string
    if return_string:
        return text

    # Otherwise, print the color-coded string
    print(text, end="")
