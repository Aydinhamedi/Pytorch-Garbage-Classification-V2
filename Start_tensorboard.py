# Libs >>>
import os
import sys
import subprocess


# Main Func >>>
def run_tensorboard():
    tensorboard_command = "tensorboard --logdir=.\\Logs\\tensorboard"

    if sys.platform.startswith("win"):
        # For Windows
        print("[Windows] Starting cmd for tensorboard...")
        subprocess.Popen(["start", "cmd", "/k", tensorboard_command], shell=True)
    elif sys.platform.startswith("darwin"):
        # For macOS
        print("[macOS] Starting Terminal for tensorboard...")
        os.system(
            f'osascript -e \'tell app "Terminal" to do script "{tensorboard_command}"\''
        )
    else:
        # For Linux
        print("[Linux] Starting gnome-terminal for tensorboard...")
        subprocess.Popen(["gnome-terminal", "--", "bash", "-c", tensorboard_command])


# Start
if __name__ == "__main__":
    run_tensorboard()
