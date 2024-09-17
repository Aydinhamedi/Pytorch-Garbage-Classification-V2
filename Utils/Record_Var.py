# Libs >>>
import copy


# Funcs & Classes >>>
class Record_var:
    def __init__(self) -> None:
        # Define the vars
        self.VarKeys = []
        # End

    def Start(self, var_scope):
        # Get the list
        self.VarKeys = copy.deepcopy(list(var_scope.keys()))
        # End

    def Capture(self, var_scope):
        # Get the new list
        New_VarKeys = var_scope.keys()
        # Get the new vars
        Added_keys = [key for key in New_VarKeys if key not in self.VarKeys]
        # End
        return {key: var_scope[key] for key in Added_keys}


# Example
if __name__ == "__main__":
    # Make the object
    var_recorder = Record_var()
    # Define some vars
    a = 1
    b = 2
    c = 4
    # Start Recording
    var_recorder.Start(globals())
    # Make some changes
    del c  # This should not be recorded
    test1 = "This should be recorded"
    test2 = "This should be recorded 2"
    # Capture the new vars
    new_vars = var_recorder.Capture(globals())
    print(new_vars)
