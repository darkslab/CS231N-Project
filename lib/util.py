from pathlib import Path
from re import search


def next_iteration_dir_path():
    iterations_dir_path = Path(__file__).resolve().parent.parent / "iterations"
    next_iteration = 1
    for iteration_path in (iterations_dir_path).iterdir():
        if search(r"\d\d\d\d", iteration_path.name) is None:
            continue
        next_iteration = max(next_iteration, int(iteration_path.name) + 1)
    return iterations_dir_path / str(next_iteration).zfill(4)
