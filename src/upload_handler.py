
from pathlib import Path
from zipfile import BadZipFile, ZipFile
import shutil


MAX_FILES = 1000
MAX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_SINGLE_FILE_BYTES = 10 * 1024 * 1024


def extract_repository_zip(
    zip_path: Path,
    destination: Path,
) -> Path:
    """Safely extract a ZIP and return the likely repository root."""
    zip_path = Path(zip_path)
    destination = Path(destination).resolve()

    if not zip_path.is_file():
        raise ValueError("The uploaded ZIP file could not be found.")

    if destination.exists():
        shutil.rmtree(destination)

    destination.mkdir(parents=True, exist_ok=True)

    try:
        with ZipFile(zip_path, "r") as archive:
            members = [
                item for item in archive.infolist()
                if not item.is_dir()
            ]

            if not members:
                raise ValueError("The ZIP file contains no files.")

            if len(members) > MAX_FILES:
                raise ValueError(
                    f"The ZIP contains too many files "
                    f"(maximum {MAX_FILES})."
                )

            total_size = sum(item.file_size for item in members)

            if total_size > MAX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    "The uncompressed ZIP is too large. "
                    "Maximum allowed size is 50 MB."
                )

            for item in members:
                if item.file_size > MAX_SINGLE_FILE_BYTES:
                    raise ValueError(
                        f"A file exceeds the 10 MB limit: "
                        f"{item.filename}"
                    )

                member_path = Path(item.filename)

                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(
                        "The ZIP contains an unsafe file path."
                    )

                # Reject symbolic links.
                unix_mode = (item.external_attr >> 16) & 0o170000
                if unix_mode == 0o120000:
                    raise ValueError(
                        "The ZIP contains a symbolic link."
                    )

                target = (destination / member_path).resolve()

                if not target.is_relative_to(destination):
                    raise ValueError(
                        "The ZIP contains a path outside its workspace."
                    )

            for item in members:
                target = (destination / item.filename).resolve()
                target.parent.mkdir(parents=True, exist_ok=True)

                with archive.open(item, "r") as source:
                    with target.open("wb") as output:
                        shutil.copyfileobj(source, output)

    except BadZipFile as exc:
        raise ValueError("This file is not a valid ZIP archive.") from exc

    # GitHub ZIPs often contain one enclosing repository directory.
    top_level_items = list(destination.iterdir())

    if len(top_level_items) == 1 and top_level_items[0].is_dir():
        repository_root = top_level_items[0]
    else:
        repository_root = destination

    python_files = list(repository_root.rglob("*.py"))

    if not python_files:
        raise ValueError(
            "No Python files were found in the uploaded repository."
        )

    return repository_root
