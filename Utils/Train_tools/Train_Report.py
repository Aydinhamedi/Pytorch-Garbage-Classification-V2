# Libs >>>
import os
import ast
import json
import inspect
import textwrap


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

    def _serialize(self, params):
        """
        Serializes the parameters in the provided dictionary into a human-readable and JSON-storable format.

        Args:
            params (dict): A dictionary of parameters to serialize.

        Returns:
            dict: A new dictionary with serialized values.
        """
        serialized_params = {}
        for key, value in params.items():
            serialized_params[key] = self._serialize_value(value, key)
        return serialized_params

    def _serialize_value(self, value, variable_name):
        """
        Serializes the value of a variable based on its type. This method is used to convert various Python objects into a serialized format that can be stored and retrieved.

        Args:
            value: The value of the variable to be serialized.
            variable_name (str): The name of the variable being serialized.

        Returns:
            dict: A dictionary containing the serialized representation of the variable, including information about its creation line and line number.
        """
        frame = (
            inspect.currentframe().f_back.f_back
        )  # Go two frames back to get the caller's frame
        try:
            source_lines, start_line = inspect.getsourcelines(frame)
        except OSError:
            return {"creation_line": "Source not available", "line_number": None}

        source = textwrap.dedent("".join(source_lines))

        tree = ast.parse(source)

        creation_line = None
        line_number = None

        class AssignmentFinder(ast.NodeVisitor):
            def visit_Assign(self, node):
                nonlocal creation_line, line_number
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == variable_name:
                        creation_line = ast.get_source_segment(source, node)
                        line_number = node.lineno + start_line - 1
                        return

        AssignmentFinder().visit(tree)

        serialized = {"creation_line": creation_line, "line_number": line_number}
        if inspect.isfunction(value) or inspect.ismethod(value):
            serialized.update(self._serialize_callable(value))
        elif inspect.ismodule(value):
            serialized.update(self._serialize_module(value))
        elif inspect.isclass(value):
            serialized.update(self._serialize_class(value))
        elif hasattr(value, "__dict__"):
            serialized.update(self._serialize_object(value))
        else:
            serialized.update(self._serialize_other(value))

        return serialized

    def _serialize_callable(self, func):
        """
        Serializes a callable (function or method) object.

        Args:
            func: The callable to serialize.

        Returns:
            dict: A dictionary containing detailed information about the callable.
        """
        try:
            source = textwrap.dedent(inspect.getsource(func))
            line = inspect.getsourcelines(func)[1]
        except (OSError, TypeError):
            source = "Source code not available"
            line = "Line number not available"

        return {
            "type": "callable",
            "name": func.__name__,
            "qualname": func.__qualname__,
            "module": func.__module__,
            "args": list(inspect.signature(func).parameters.keys()),
            "annotations": {k: str(v) for k, v in func.__annotations__.items()},
            "defaults": [str(v) for v in (func.__defaults__ or [])],
            "doc": inspect.getdoc(func),
            "source": source,
            "line": line,
        }

    def _serialize_module(self, module):
        """
        Serializes a module object.

        Args:
            module: The module to serialize.

        Returns:
            dict: A dictionary containing detailed information about the module.
        """
        try:
            source = textwrap.dedent(inspect.getsource(module))
        except (OSError, TypeError):
            source = "Source code not available"

        return {
            "type": "module",
            "name": module.__name__,
            "doc": inspect.getdoc(module),
            "file": getattr(module, "__file__", "File not available"),
            "members": {
                name: type(member).__name__
                for name, member in inspect.getmembers(module)
            },
            "version": getattr(module, "__version__", "Version not available"),
            "source": source,
        }

    def _serialize_class(self, cls):
        """
        Serializes a class object.

        Args:
            cls: The class to serialize.

        Returns:
            dict: A dictionary containing detailed information about the class.
        """
        try:
            source = textwrap.dedent(inspect.getsource(cls))
        except (OSError, TypeError):
            source = "Source code not available"

        return {
            "type": "class",
            "name": cls.__name__,
            "module": cls.__module__,
            "doc": inspect.getdoc(cls),
            "methods": self._serialize_methods(cls),
            "attributes": {
                k: str(v) for k, v in cls.__dict__.items() if not callable(v)
            },
            "base_classes": [base.__name__ for base in cls.__bases__],
            "source": source,
        }

    def _serialize_object(self, obj):
        """
        Serializes a general object.

        Args:
            obj: The object to serialize.

        Returns:
            dict: A dictionary containing detailed information about the object.
        """
        try:
            source = textwrap.dedent(inspect.getsource(obj.__class__))
        except (OSError, TypeError):
            source = "Source code not available"

        return {
            "type": "object",
            "class": obj.__class__.__name__,
            "module": obj.__class__.__module__,
            "attributes": {k: str(v) for k, v in obj.__dict__.items()},
            "source": source,
        }

    def _serialize_methods(self, cls):
        """
        Serializes the methods of a class.

        Args:
            cls: The class whose methods to serialize.

        Returns:
            dict: A dictionary containing detailed information about each method.
        """
        methods = {}
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            methods[name] = self._serialize_callable(method)
        return methods

    def _serialize_other(self, value):
        """
        Serializes other types of values.

        Args:
            value: The value to serialize.

        Returns:
            dict: A dictionary containing basic information about the value.
        """
        return {
            "type": type(value).__name__,
            "value": repr(value),
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

    def Del_Report(self):
        """Deletes the training report, including the training parameters and history."""
        del self
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
