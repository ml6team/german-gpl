from pathlib import Path

def create_dir(path):
    """
    Create new dir. 
    """
    Path(path).mkdir(parents=True, exist_ok=True)

