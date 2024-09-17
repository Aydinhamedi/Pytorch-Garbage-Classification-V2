# Libs >>>
import os
import json
import inspect


# Funcs & Classes >>>
class Reporter:
    def __init__(self, save_dir):
        """
        Initializes a Reporter object with a specified save directory.

        The Reporter class is used to manage the training history and parameters for a machine learning model. The __init__ method sets up the initial state of the Reporter object, including the save directory, training history, and training parameters.

        Args:
            save_dir (str): The directory where the training report will be saved.
        """
        # Define the vars
        self.save_dir = save_dir
        self.train_history = []
        self.train_prams = {}
        # Make the save dir
        os.makedirs(self.save_dir, exist_ok=True)
        # End

    def _serialize_callables(self, prams):
        """
        Serializes the callable parameters in the provided dictionary `prams` by extracting their name, arguments, docstring, source code location, and usage in the main code. This is used to save the training parameters in a format that can be easily inspected and restored later.

        Args:
            prams (dict): A dictionary of training parameters, where the values may be callable objects (e.g. functions).

        Returns:
            dict: A new dictionary with the same keys as `prams`, but the callable values are replaced with a dictionary containing the function's metadata.
        """
        serialized_prams = {}

        for key, value in prams.items():
            if callable(value):
                func_name = value.__name__
                func_args = inspect.signature(value).parameters
                func_doc = inspect.getdoc(value)
                try:
                    func_source = inspect.getsource(value)
                    func_file = inspect.getfile(value)
                    func_line = inspect.getsourcelines(value)[1]
                except (OSError, TypeError):
                    func_source = "Source code not available"
                    func_file = "File not available"
                    func_line = "Line number not available"

                # Find the usage of the function in the main code
                usage = f"{func_name}({', '.join(func_args.keys())})"

                serialized_prams[key] = {
                    "type": "function",
                    "name": func_name,
                    "args": list(func_args.keys()),
                    "doc": func_doc,
                    "source": func_source,
                    "file": func_file,
                    "line": func_line,
                    "usage": usage,
                    "module": value.__module__,
                }
            else:
                serialized_prams[key] = value

        return serialized_prams

    def Add_Prams(self, prams):
        """Adds the training parameters to the report, serializing any callable parameters (e.g. functions) to capture their metadata.

        Args:
            prams (dict): A dictionary of training parameters, where the values may be callable objects (e.g. functions)."""
        # Add the prams
        self.train_prams = self._serialize_callables(prams)
        # End

    def Add_History(self, history):
        """Adds the training history to the report.

        Args:
            history (dict): A dictionary containing the training history.
        """
        # Add the history
        self.train_history = history
        # End

    def Save_Report(self):
        """
        Saves the training report, including the training parameters and history, to JSON files in the specified save directory.

        Raises:
            ValueError: If the training history or parameters have not been set using the `Add_History` or `Add_Prams` methods.
        """
        # Check if the history is empty
        if len(self.train_history) == 0:
            raise ValueError(
                "The history is empty: Add the history using the Add_History method."
            )
        # Check if the prams are empty
        if len(self.train_prams) == 0:
            raise ValueError(
                "The prams are empty: Add the prams using the Add_Prams method."
            )
        # Save the prams in a JSON file
        with open(os.path.join(self.save_dir, "prams.json"), "w") as f:
            json.dump(self.train_prams, f, indent=4)
        # Save the history in a JSON file
        with open(os.path.join(self.save_dir, "history.json"), "w") as f:
            json.dump(self.train_history, f, indent=4)
        # End
