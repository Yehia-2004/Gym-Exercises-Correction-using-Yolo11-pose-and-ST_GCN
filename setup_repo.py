from pathlib import Path
import subprocess
import sys

def setup(repo_url: Path):
    repo_path = Path(__file__).parent / repo_url.stem
    if not repo_path.exists():
        subprocess.run(["git", "clone", repo_url])
    
    sys.path.append(str(repo_path.absolute()))