
from pathlib import Path

from src.code_chunker import (
    CodeChunk,
    chunk_python_file,
    find_section_headings,
    is_section_heading,
    remove_exact_duplicate_chunks,
)

from src.code_chunker import (
    DEFAULT_REPOSITORY,
    chunk_repository,
)
from src.repository_loader import find_python_files


def test_recognizes_uppercase_section_heading():
    """Uppercase section headings should be recognized."""

    assert is_section_heading("# PATHS (YOUR ACTUAL DATA)") is True


def test_recognizes_unicode_separator_heading():
    """Decorative Unicode separators should not hide a heading."""

    heading = "# ── DATA PATHS ─────────────────────────"

    assert is_section_heading(heading) is True


def test_ignores_ordinary_comment():
    """An ordinary explanatory comment should not become a heading."""

    assert is_section_heading("# Freeze base layers") is False


def test_extracts_clean_section_title():
    """The section title should retain meaningful punctuation."""

    source_lines = [
        "# =========================",
        "# 📌 PATHS (YOUR ACTUAL DATA)",
        "# =========================",
        'train_dir = "data/train"',
    ]

    headings = find_section_headings(source_lines)

    assert len(headings) == 1
    assert headings[0][1] == "PATHS (YOUR ACTUAL DATA)"


def test_chunk_ranges_are_valid(tmp_path: Path):
    """Every produced chunk should have a valid line range and source."""

    repository = tmp_path / "repository"
    repository.mkdir()

    source_file = repository / "example.py"
    source_file.write_text(
        'import os\n'
        '\n'
        '# DATA PATHS\n'
        'train_dir = "data/train"\n'
        'test_dir = "data/test"\n',
        encoding="utf-8",
    )

    chunks = chunk_python_file(source_file, repository)

    assert chunks, "Expected at least one chunk."

    source_lines = source_file.read_text(encoding="utf-8").splitlines()

    for chunk in chunks:
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line
        assert chunk.end_line <= len(source_lines)
        assert chunk.code.strip()


def test_removes_exact_duplicate_records():
    """An identical chunk record should appear only once."""

    chunk = CodeChunk(
        file_path="src/example.py",
        symbol_name="example",
        start_line=1,
        end_line=2,
        code="x = 1\ny = 2",
        chunk_type="module",
    )

    result = remove_exact_duplicate_chunks([chunk, chunk])

    assert len(result) == 1


def test_preserves_identical_code_at_different_locations():
    """Different source locations must retain their citation metadata."""

    first = CodeChunk(
        file_path="src/example.py",
        symbol_name="first",
        start_line=1,
        end_line=1,
        code="value = 10",
        chunk_type="module",
    )

    second = CodeChunk(
        file_path="src/example.py",
        symbol_name="second",
        start_line=5,
        end_line=5,
        code="value = 10",
        chunk_type="module",
    )

    result = remove_exact_duplicate_chunks([first, second])

    assert len(result) == 2
    assert result[0].start_line == 1
    assert result[1].start_line == 5


def test_real_repository_has_python_files():
    """The configured repository should contain Python source files."""

    python_files = find_python_files(DEFAULT_REPOSITORY)

    assert python_files, "No Python files were discovered."


def test_real_repository_produces_chunks():
    """The real repository should produce non-empty chunks."""

    chunks = chunk_repository(DEFAULT_REPOSITORY)

    assert chunks, "The chunker produced no chunks."

    for chunk in chunks:
        assert chunk.file_path
        assert chunk.code.strip()
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line


def test_real_repository_has_no_exact_duplicate_records():
    """The final chunk list should contain no identical chunk records."""

    chunks = chunk_repository(DEFAULT_REPOSITORY)

    identities = [
        (
            chunk.file_path,
            chunk.symbol_name,
            chunk.start_line,
            chunk.end_line,
            chunk.chunk_type,
            chunk.code,
        )
        for chunk in chunks
    ]

    assert len(identities) == len(set(identities))

