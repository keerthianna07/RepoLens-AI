
import ast
from pathlib import Path

from src.repository_loader import DEFAULT_REPOSITORY, find_python_files


MODEL_NAMES = {
    "EfficientNetB0",
    "EfficientNetB3",
    "ResNet50",
    "VGG19",
    "MobileNetV3Large",
    "MobileNetV3Small",
    "DenseNet121",
    "InceptionV3",
}


def find_model_architectures(repository_path: Path) -> list[dict]:
    """Find recognized model architecture references without executing code."""

    discovered = {}

    for file_path in find_python_files(repository_path):
        source = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            model_name = None
            line_number = getattr(node, "lineno", None)

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    model_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    model_name = node.func.attr

            elif isinstance(node, ast.Name):
                model_name = node.id

            if model_name in MODEL_NAMES:
                relative_path = str(
                    file_path.relative_to(repository_path)
                )

                key = (model_name, relative_path)

                if key not in discovered:
                    discovered[key] = {
                        "model": model_name,
                        "file": relative_path,
                        "line": line_number,
                    }

    return sorted(
        discovered.values(),
        key=lambda item: (item["model"], item["file"]),
    )


def main() -> None:
    """Print recognized model references in the repository."""

    results = find_model_architectures(DEFAULT_REPOSITORY)

    print("Recognized model architecture references:")
    print(f"Total references found: {len(results)}")

    for item in results:
        print(
            f"- {item['model']} | "
            f"{item['file']} | "
            f"line {item['line']}"
        )


if __name__ == "__main__":
    main()
