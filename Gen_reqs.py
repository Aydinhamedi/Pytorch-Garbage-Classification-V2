# Libs >>>
import os
import subprocess


# Main func >>>
def run_pipreqs():
    # Ensure Logs\others directory exists
    if not os.path.exists("Logs\\others"):
        os.makedirs("Logs\\others")

    # Define the log file path
    log_file_path = os.path.join("Logs\\others", "pipreqs_log.txt")

    # Define the pipreqs command with parameters
    command = [
        "pipreqs",
        "--ignore",
        ".venv",
        "--encoding=utf-8",
        "--scan-notebooks",
        "--force",
        "--mode",
        "compat",
        "--debug",
        ".",
    ]

    # Run the command and capture output
    with open(log_file_path, "w") as log_file:
        result = subprocess.run(command, capture_output=True, text=True)
        log_file.write(result.stdout)
        if result.stderr:
            log_file.write(result.stderr)

    # Print success or error message
    if result.returncode == 0:
        print("Requirements file generated successfully.")
    else:
        print(
            f"An error occurred. Please check the log file for more details: {log_file_path}"
        )


if __name__ == "__main__":
    run_pipreqs()
