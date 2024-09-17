# Libs >>>
import os
import json
import types
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

    def _serialize(self, prams):
        """
        Serializes the callable parameters, objects, and modules in the provided dictionary.

        Args:
            prams (dict): A dictionary of training parameters.

        Returns:
            dict: A new dictionary with serialized values.
        """
        serialized_prams = {}

        for key, value in prams.items():
            if callable(value):
                serialized_prams[key] = self._serialize_callable(value)
            elif isinstance(value, types.ModuleType):
                serialized_prams[key] = self._serialize_module(value)
            elif hasattr(value, "__dict__"):
                serialized_prams[key] = self._serialize_object(value)
            else:
                serialized_prams[key] = self._serialize_other(value)

        return serialized_prams

    def _serialize_callable(self, func):
        """Serialize a callable object."""
        func_name = func.__name__
        func_args = inspect.signature(func).parameters
        func_doc = inspect.getdoc(func)
        
        try:
            func_source = inspect.getsource(func)
            func_file = inspect.getfile(func)
            func_line = inspect.getsourcelines(func)[1]
        except (OSError, TypeError):
            func_source = "Source code not available"
            func_file = "File not available"
            func_line = "Line number not available"

        usage = f"{func_name}({', '.join(func_args.keys())})"

        return {
            "type": "function",
            "name": func_name,
            "args": list(func_args.keys()),
            "doc": func_doc,
            "source": func_source,
            "file": func_file,
            "line": func_line,
            "usage": usage,
            "module": func.__module__,
            "return_annotation": str(inspect.signature(func).return_annotation),
        }

    def _serialize_module(self, module):
        """Serialize a module object."""
        return {
            "type": "module",
            "name": module.__name__,
            "doc": inspect.getdoc(module),
            "file": getattr(module, "__file__", "File not available"),
            "members": {name: str(member) for name, member in inspect.getmembers(module)},
            "version": getattr(module, "__version__", "Version not available"),
        }

    def _serialize_object(self, obj):
        """Serialize a general object."""
        obj_class = obj.__class__
        return {
            "type": "object",
            "class": obj_class.__name__,
            "module": obj.__module__,
            "doc": inspect.getdoc(obj_class),
            "methods": self._serialize_methods(obj_class),
            "attributes": {k: str(v) for k, v in obj.__dict__.items()},
            "base_classes": [base.__name__ for base in obj_class.__bases__],
        }

    def _serialize_methods(self, cls):
        """Serialize methods of a class."""
        methods = {}
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            methods[name] = {
                "name": method.__name__,
                "args": list(inspect.signature(method).parameters.keys()),
                "doc": inspect.getdoc(method),
                "source": inspect.getsource(method) if inspect.isfunction(method) else "Source code not available",
                "file": inspect.getfile(method) if inspect.isfunction(method) else "File not available",
                "line": inspect.getsourcelines(method)[1] if inspect.isfunction(method) else "Line number not available",
                "return_annotation": str(inspect.signature(method).return_annotation),
            }
        return methods

    def _serialize_other(self, value):
        """Serialize other types of values."""
        return {
            "type": type(value).__name__,
            "value": str(value),
            "repr": repr(value),
        }


    def Add_Prams(self, prams):
        """Adds the training parameters to the report, serializing any callable parameters (e.g. functions) to capture their metadata.

        Args:
            prams (dict): A dictionary of training parameters, where the values may be callable objects (e.g. functions)."""
        # Add the prams
        self.train_prams = self._serialize(prams)
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



