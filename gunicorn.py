# gunicorn_config.py

import os
import sys

def post_fork(server, worker):
    # Adding the project directory to sys.path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_dir)

    # Importing the function
    global tokenize_kh, remove_affixes
    from preprocessing import tokenize_kh, remove_affixes

    # Log to ensure the function is imported
    server.log.info("tokenize_kh function imported successfully in worker.")

bind = "0.0.0.0:8000"
workers = 4
threads = 2
timeout = 120

# Adding the post_fork hook
post_fork = post_fork
