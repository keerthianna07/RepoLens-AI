
from pathlib import Path


DEFAULT_REPOSITORY = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "repository"
    / "knee-osteoarthritis-classification"
)

IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "site-packages",
    "dist",
    "build",
}

IGNORED_FILE_NAMES = {
    ".env",
    "secrets.py",
}

IGNORED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".log",
}


def find_python_files(repository_path: Path) -> list[Path]:
    """Return Python source files while excluding unwanted paths."""
    if not repository_path.exists():
        raise FileNotFoundError(
            f"Repository directory does not exist: {repository_path}"
        )

    if not repository_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, received: {repository_path}"
        )

    python_files = []

    for file_path in repository_path.rglob("*.py"):
        relative_path = file_path.relative_to(repository_path)

        if any(
            part in IGNORED_DIRECTORIES
            for part in relative_path.parts
        ):
            continue

        if file_path.name in IGNORED_FILE_NAMES:
            continue

        if file_path.suffix.lower() in IGNORED_SUFFIXES:
            continue

        if not file_path.is_file():
            continue

        python_files.append(file_path)

    return sorted(python_files)


def print_repository_summary(repository_path: Path) -> None:
    """Print the discovered Python files and their sizes."""
    python_files = find_python_files(repository_path)

    print(f"Repository: {repository_path}")
    print(f"Python files found: {len(python_files)}")
    print("\nDiscovered files:")

    total_bytes = 0

    for file_path in python_files:
        relative_path = file_path.relative_to(repository_path)
        file_size = file_path.stat().st_size
        total_bytes += file_size

        print(f"  {relative_path} ({file_size:,} bytes)")

    print(f"\nTotal Python source size: {total_bytes:,} bytes")


if __name__ == "__main__":
    print_repository_summary(DEFAULT_REPOSITORY)
