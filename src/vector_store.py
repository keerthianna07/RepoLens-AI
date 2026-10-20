
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.code_chunker import chunk_repository
from src.repository_loader import DEFAULT_REPOSITORY

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTOR_STORE_PATH = PROJECT_ROOT / "storage" / "chroma_db"

COLLECTION_NAME = "repolens_code"



def create_documents(
    repository_path: Path = DEFAULT_REPOSITORY,
) -> list[Document]:
    """Convert code chunks into documents with source metadata."""
    chunks = chunk_repository(repository_path)
    documents = []

    for chunk in chunks:
        document = Document(
            page_content=chunk.code,
            metadata={
                "file_path": str(chunk.file_path),
                "symbol_name": chunk.symbol_name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_type": chunk.chunk_type,
            },
        )
        documents.append(document)

    return documents




def build_vector_store(
    repository_path: Path = DEFAULT_REPOSITORY,
    vector_store_path: Path = VECTOR_STORE_PATH,
) -> Chroma:
    """Embed repository code chunks and store their vectors in ChromaDB."""
    repository_path = Path(repository_path)
    vector_store_path = Path(vector_store_path)

    documents = create_documents(repository_path)

    if not documents:
        raise ValueError(
            f"No code chunks found in {repository_path}"
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # Rebuild only the vector-store directory selected for this run.
    if vector_store_path.exists():
        shutil.rmtree(vector_store_path)

    vector_store_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(vector_store_path),
    )

    vector_store.add_documents(documents)

    print("Vector indexing completed successfully.")
    print(f"Repository indexed: {repository_path}")
    print(f"Code chunks indexed: {len(documents)}")
    print(f"Vector store location: {vector_store_path}")

    return vector_store


def search_similar_code(
    vector_store: Chroma,
    query: str,
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """Find semantically similar code chunks for a user question."""

    if not query.strip():
        return []

    if top_k < 1:
        return []

    return vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
    )


def main() -> None:
    """Build the vector index and run a semantic search."""

    print("Building the RepoLens AI semantic index...")

    vector_store = build_vector_store()

    while True:
        query = input(
            "\nAsk a question about the repository (or type 'exit'): "
        ).strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting RepoLens AI.")
            break

        if not query:
            print("Please enter a question.")
            continue

        results = search_similar_code(
            vector_store,
            query,
            top_k=5,
        )

        if not results:
            print("No results returned.")
            continue

        print(f"\nTop {len(results)} semantic search results:")

        for rank, (document, distance) in enumerate(
            results,
            start=1,
        ):
            metadata = document.metadata

            print(f"\n[{rank}] Distance: {distance:.4f}")
            print(f"File: {metadata['file_path']}")
            print(
                f"Lines: {metadata['start_line']}-"
                f"{metadata['end_line']}"
            )
            print(f"Symbol: {metadata['symbol_name']}")
            print("-" * 60)

            preview = document.page_content[:700]
            print(preview)

            if len(document.page_content) > 700:
                print("\n... [preview truncated]")


if __name__ == "__main__":
    main()
