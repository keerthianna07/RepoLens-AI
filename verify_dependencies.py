
import importlib
import sys

PACKAGES = {
    "langchain": "langchain",
    "langchain_text_splitters": "langchain_text_splitters",
    "langchain_google_genai": "langchain_google_genai",
    "langchain_chroma": "langchain_chroma",
    "langgraph": "langgraph",
    "rank_bm25": "rank_bm25",
    "streamlit": "streamlit",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "dotenv": "dotenv",
    "pydantic_settings": "pydantic_settings",
    "pandas": "pandas",
    "pytest": "pytest",
    "chromadb": "chromadb",
}


def main():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print("\nChecking project dependencies...\n")

    failures = []

    for package_name, import_name in PACKAGES.items():
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "version not exposed")
            print(f"PASS  {package_name}: {version}")
        except Exception as error:
            failures.append(package_name)
            print(
                f"FAIL  {package_name}: "
                f"{type(error).__name__}: {error}"
            )

    print("\n" + "=" * 45)

    if failures:
        print("Dependency verification FAILED.")
        print("Packages needing attention:", ", ".join(failures))
        raise SystemExit(1)

    print("SUCCESS: All required imports are working.")


if __name__ == "__main__":
    main()
