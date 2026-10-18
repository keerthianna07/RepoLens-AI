
import ast
import re
from dataclasses import dataclass
from pathlib import Path

from src.repository_loader import DEFAULT_REPOSITORY, find_python_files


# Ordinary module-level chunks should stay reasonably small.
MAX_MODULE_CHUNK_LINES = 35


@dataclass
class CodeChunk:
    """A piece of source code with metadata for retrieval and citations."""

    file_path: str
    symbol_name: str
    start_line: int
    end_line: int
    code: str
    chunk_type: str


def extract_source_segment(
    source_lines: list[str],
    start_line: int,
    end_line: int,
) -> str:
    """Extract source lines using human-readable, 1-based line numbers."""

    return "\n".join(source_lines[start_line - 1:end_line])


def make_chunk(
    source_lines: list[str],
    relative_path: str,
    symbol_name: str,
    start_line: int,
    end_line: int,
    chunk_type: str,
) -> CodeChunk | None:
    """Create a chunk if its line range and source code are valid."""

    if start_line < 1 or end_line < start_line:
        return None

    if end_line > len(source_lines):
        return None

    code = extract_source_segment(
        source_lines,
        start_line,
        end_line,
    )

    if not code.strip():
        return None

    return CodeChunk(
        file_path=relative_path,
        symbol_name=symbol_name,
        start_line=start_line,
        end_line=end_line,
        code=code,
        chunk_type=chunk_type,
    )


def chunk_definitions(
    tree: ast.AST,
    source_lines: list[str],
    relative_path: str,
) -> list[CodeChunk]:
    """Create separate chunks for functions, async functions, and classes."""

    chunks = []
    seen_definitions = set()

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            continue

        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)

        if start_line is None or end_line is None:
            continue

        if isinstance(node, ast.ClassDef):
            chunk_type = "class"
        elif isinstance(node, ast.AsyncFunctionDef):
            chunk_type = "async_function"
        else:
            chunk_type = "function"

        # The same AST definition should only be recorded once.
        identity = (
            relative_path,
            node.name,
            start_line,
            end_line,
            chunk_type,
        )

        if identity in seen_definitions:
            continue

        seen_definitions.add(identity)

        chunk = make_chunk(
            source_lines=source_lines,
            relative_path=relative_path,
            symbol_name=node.name,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
        )

        if chunk is not None:
            chunks.append(chunk)

    return chunks


def is_definition(node: ast.AST) -> bool:
    """Return True for top-level function and class definitions."""

    return isinstance(
        node,
        (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        ),
    )


def is_import(node: ast.AST) -> bool:
    """Return True for Python import statements."""

    return isinstance(node, (ast.Import, ast.ImportFrom))


def is_separator_comment(line: str) -> bool:
    """Recognize decorative comment lines such as '# ======='. """

    stripped = line.strip()

    if not stripped.startswith("#"):
        return False

    return re.fullmatch(r"[#\s=*_~\-─━]+", stripped) is not None




def is_section_heading(line: str) -> bool:
    """Recognize uppercase headings and headings with decorative separators."""

    stripped = line.strip()

    if not stripped.startswith("#"):
        return False

    content = stripped[1:].strip()

    if not content or is_separator_comment(stripped):
        return False

    # A decorative line such as '# ── PHASE 1: Feature Extraction ──'
    # is a heading even when its words use mixed capitalization.
    has_unicode_separator = bool(
        re.match(r"^[\s─━]+", content)
    )

    # Remove decorative symbols only for checking the heading format.
    cleaned = re.sub(r"^[\W_]+", "", content).strip()
    letters = re.findall(r"[A-Za-z]", cleaned)

    if len(letters) < 3:
        return False

    if has_unicode_separator:
        return True

    uppercase_ratio = (
        sum(letter.isupper() for letter in letters) / len(letters)
    )

    return uppercase_ratio >= 0.80

def find_section_headings(
    source_lines: list[str],
) -> list[tuple[int, str]]:
    """
    Find section headings and return (section_start_line, heading_title).

    If a decorative separator appears immediately before a heading,
    the section starts at the separator.
    """

    headings = []

    for index, line in enumerate(source_lines):
        if not is_section_heading(line):
            continue

        heading_title = line.strip()[1:].strip()

        # Remove decorative symbols at the beginning, but preserve meaningful
        # punctuation such as the closing parenthesis in a heading.
        heading_title = re.sub(r"^[\W_]+", "", heading_title).strip()

        # Remove trailing decorative separator characters only.
        heading_title = re.sub(r"[\s─━=_*~\-]+$", "", heading_title).strip()
        heading_start_line = index + 1

        # Include a decorative separator directly above the heading.
        if index > 0 and is_separator_comment(source_lines[index - 1]):
            heading_start_line = index

        headings.append((heading_start_line, heading_title))

    return headings


def get_section_for_line(
    line_number: int,
    section_headings: list[tuple[int, str]],
) -> tuple[int, str | None, int | None]:
    """
    Find the most recent section heading at or before a source line.

    Returns:
        (section_index, heading_title, section_start_line)

    Lines before the first heading belong to the preamble section.
    """

    selected_index = -1
    selected_title = None
    selected_start = None

    for index, (start_line, title) in enumerate(section_headings):
        if start_line > line_number:
            break

        selected_index = index
        selected_title = title
        selected_start = start_line

    return selected_index, selected_title, selected_start



def describe_chunk(
    section_title: str | None,
    code: str,
    chunk_type: str,
) -> str:
    """Describe a chunk using its section and recognizable code patterns."""

    if chunk_type == "import":
        return "module_imports"

    text = code.lower()
    section = section_title or "module-level code"

    # Detect the actual operation before checking broad variable names.
    has_model_fit = "model.fit(" in text
    has_model_compile = "model.compile(" in text
    has_model_architecture = any(
        name in text
        for name in (
            "efficientnetb0(",
            "efficientnetb3(",
            "resnet50(",
            "vgg19(",
        )
    )

    if has_model_fit:
        detail = "model training"

    elif has_model_architecture and has_model_compile:
        detail = "model architecture and compilation"

    elif has_model_architecture:
        detail = "model architecture"

    elif has_model_compile:
        detail = "model compilation"

    elif "modelcheckpoint(" in text:
        detail = "model checkpoint configuration"

    elif "earlystopping(" in text or "reducelronplateau(" in text:
        detail = "training callbacks"

    elif "compute_class_weight(" in text:
        detail = "class weight calculation"

    elif "test_gen =" in text and "flow_from_directory(" in text:
        detail = "test data generator"

    elif "val_gen =" in text and "flow_from_directory(" in text:
        detail = "validation data generator"

    elif "train_gen =" in text and "flow_from_directory(" in text:
        detail = "training data generator"

    elif "train_datagen =" in text:
        detail = "training data augmentation"

    elif "val_test_datagen =" in text:
        detail = "validation and test preprocessing"

    elif re.search(r"^\s*(train_dir|val_dir|test_dir)\s*=", text, re.M):
        detail = "dataset paths"

    elif "confusion_matrix(" in text or "classification_report(" in text:
        detail = "classification evaluation"

    elif "f1_score(" in text or "precision_score(" in text:
        detail = "evaluation metrics"

    elif "model.predict(" in text:
        detail = "model predictions"

    elif "plt." in text:
        detail = "plots and visualizations"

    else:
        return f"section: {section}"

    return f"{section} — {detail}"

def group_module_statements(
    tree: ast.AST,
    source_lines: list[str],
    relative_path: str,
) -> list[CodeChunk]:
    """
    Group top-level statements into section-aware module chunks.

    Chunks are split when:
    - a class or function definition is encountered;
    - the import/module statement category changes;
    - a new recognized section begins; or
    - the chunk exceeds the preferred line limit.
    """

    chunks = []
    section_headings = find_section_headings(source_lines)

    current_nodes = []
    current_category = None
    current_section_index = None
    current_section_title = None
    current_section_start = None

    # Track which sections have already included their heading in a chunk.
    sections_with_heading = set()

    def flush_group() -> None:
        nonlocal current_nodes
        nonlocal current_category
        nonlocal current_section_index
        nonlocal current_section_title
        nonlocal current_section_start

        if not current_nodes:
            return

        first_node_line = min(node.lineno for node in current_nodes)
        last_node_line = max(node.end_lineno for node in current_nodes)

        if current_category == "import":
            chunk_type = "import"
        else:
            chunk_type = "module"

        if chunk_type == "import":
            symbol_name = "module_imports"
        elif current_section_title:
            symbol_name = f"section: {current_section_title}"
        else:
            symbol_name = "module_level_block"

        # Include a section heading only in the first module chunk
        # created for that section.
        start_line = first_node_line

        if (
            current_section_index is not None
            and current_section_start is not None
            and current_section_index not in sections_with_heading
        ):
            start_line = min(current_section_start, first_node_line)
            sections_with_heading.add(current_section_index)

        chunk = make_chunk(
            source_lines=source_lines,
            relative_path=relative_path,
            symbol_name=symbol_name,
            start_line=start_line,
            end_line=last_node_line,
            chunk_type=chunk_type,
        )

        if chunk is not None:
            chunk.symbol_name = describe_chunk(
                section_title=current_section_title,
                code=chunk.code,
                chunk_type=chunk.chunk_type,
            )
            chunks.append(chunk)

        current_nodes = []
        current_category = None
        current_section_index = None
        current_section_title = None
        current_section_start = None

    for node in tree.body:
        # Definitions receive their own chunks, not module-level chunks.
        if is_definition(node):
            flush_group()
            continue

        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)

        if start_line is None or end_line is None:
            flush_group()
            continue

        section_index, section_title, section_start = get_section_for_line(
            start_line,
            section_headings,
        )

        category = "import" if is_import(node) else "module"

        # Do not combine statements across section or category boundaries.
        section_changed = (
            current_nodes
            and section_index != current_section_index
        )

        category_changed = (
            current_nodes
            and category != current_category
        )

        if section_changed or category_changed:
            flush_group()

        if not current_nodes:
            current_category = category
            current_section_index = section_index
            current_section_title = section_title
            current_section_start = section_start

        proposed_start = min(
            [existing.lineno for existing in current_nodes] + [start_line]
        )

        proposed_end = max(
            [existing.end_lineno for existing in current_nodes] + [end_line]
        )

        proposed_size = proposed_end - proposed_start + 1

        # Keep a normal module chunk under the preferred size where possible.
        # A single oversized statement is kept intact.
        if current_nodes and proposed_size > MAX_MODULE_CHUNK_LINES:
            flush_group()

            current_category = category
            current_section_index = section_index
            current_section_title = section_title
            current_section_start = section_start

        current_nodes.append(node)

    flush_group()

    return chunks


def remove_exact_duplicate_chunks(
    chunks: list[CodeChunk],
) -> list[CodeChunk]:
    """
    Remove duplicate chunk records without losing source provenance.

    Identical code at different file paths or line ranges is preserved.
    Parent/child chunks, such as a class and its method, may overlap
    intentionally and are also preserved.
    """

    unique_chunks = []
    seen_chunks = set()

    for chunk in chunks:
        identity = (
            chunk.file_path,
            chunk.symbol_name,
            chunk.start_line,
            chunk.end_line,
            chunk.chunk_type,
            chunk.code,
        )

        if identity in seen_chunks:
            continue

        seen_chunks.add(identity)
        unique_chunks.append(chunk)

    return unique_chunks


def chunk_python_file(
    file_path: Path,
    repository_path: Path,
) -> list[CodeChunk]:
    """Read and chunk one Python file without executing its code."""

    try:
        source = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        print(f"Warning: Could not read {file_path}: {error}")
        return []

    source_lines = source.splitlines()

    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as error:
        print(f"Warning: Could not parse {file_path}: {error}")
        return []

    relative_path = file_path.relative_to(repository_path).as_posix()

    definition_chunks = chunk_definitions(
        tree=tree,
        source_lines=source_lines,
        relative_path=relative_path,
    )

    module_chunks = group_module_statements(
        tree=tree,
        source_lines=source_lines,
        relative_path=relative_path,
    )

    all_chunks = definition_chunks + module_chunks

    unique_chunks = remove_exact_duplicate_chunks(all_chunks)

    return sorted(
        unique_chunks,
        key=lambda chunk: (
            chunk.file_path,
            chunk.start_line,
            chunk.end_line,
            chunk.chunk_type,
            chunk.symbol_name,
        ),
    )


def chunk_repository(repository_path: Path) -> list[CodeChunk]:
    """Create chunks from every Python file found in the repository."""

    all_chunks = []

    python_files = find_python_files(repository_path)

    for file_path in python_files:
        all_chunks.extend(
            chunk_python_file(
                file_path=file_path,
                repository_path=repository_path,
            )
        )

    return remove_exact_duplicate_chunks(all_chunks)


def main() -> None:
    """Run the chunker and print a readable summary."""

    chunks = chunk_repository(DEFAULT_REPOSITORY)

    print(f"Repository: {DEFAULT_REPOSITORY}")
    print(f"Total code chunks: {len(chunks)}")

    chunk_counts = {}

    for chunk in chunks:
        chunk_counts[chunk.chunk_type] = (
            chunk_counts.get(chunk.chunk_type, 0) + 1
        )

    print("\nChunk counts by type:")

    for chunk_type, count in sorted(chunk_counts.items()):
        print(f"  {chunk_type}: {count}")

    print("\nDiscovered chunks:")

    for chunk in chunks:
        preview = chunk.code[:180].replace("\n", " ")

        print(
            f"\n[{chunk.chunk_type}] "
            f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
        )
        print(f"Symbol: {chunk.symbol_name}")
        print(f"Preview: {preview}")


if __name__ == "__main__":
    main()
