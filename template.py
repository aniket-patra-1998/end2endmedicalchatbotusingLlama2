import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_file = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "research/trials.ipynb",
    "app.py",
    "store_index.py",
    "static/.gitkeep",
    'templates/chat.html'
]

for file_path in list_of_file:
    filepath = Path(file_path)
    filedir,filename = os.path.split(filepath)
    if filedir !="":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"creating file directory; {filedir} for the file {filename}")
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f"creating empty file : {filepath}")
    else:
        logging.info(f"{filename} is already created")


