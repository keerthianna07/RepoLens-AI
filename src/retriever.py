
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.code_chunker import CodeChunk, chunk_repository
from src.repository_loader import DEFAULT_REPOSITORY


@dataclass
class SearchResult:
    """One retrieved code chunk and its relevance score."""

    chunk: CodeChunk
    score: float


def tokenize(text: str) -> list[str]:
    """Split natural-language questions and code into searchable tokens."""

    raw_tokens = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*|\d+",
        text.lower(),
    )

    tokens = []

    for token in raw_tokens:
        tokens.append(token)

        # Make identifiers such as flow_from_directory
        # searchable using their individual words too.
        parts = token.split("_")

        for part in parts:
            if part and part != token:
                tokens.append(part)

    return tokens


class BM25Retriever:
    """Retrieve relevant source-code chunks using keyword relevance."""

    def __init__(self, chunks: list[CodeChunk]) -> None:
        if not chunks:
            raise ValueError(
                "No code chunks were provided to the retriever."
            )

        self.chunks = chunks

        # Include code, symbol names and file paths in the search index.
        self.tokenized_corpus = [
            tokenize(
                f"{chunk.file_path} "
                f"{chunk.symbol_name} "
                f"{chunk.code}"
            )
            for chunk in chunks
        ]

        self.index = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return up to top_k relevant chunks for a question."""

        if not query.strip():
            return []

        if top_k < 1:
            return []

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            # Do not return chunks with no keyword relevance.
            if score <= 0:
                continue

            results.append(
                SearchResult(
                    chunk=self.chunks[index],
                    score=score,
                )
            )

            if len(results) == top_k:
                break

        return results


def build_retriever(
    repository_path: Path = DEFAULT_REPOSITORY,
) -> BM25Retriever:
    """Read the repository, create chunks and build the search index."""

    chunks = chunk_repository(repository_path)

    if not chunks:
        raise ValueError(
            f"No code chunks found in: {repository_path}"
        )

    return BM25Retriever(chunks)


def main() -> None:
    """Run a simple interactive command-line search."""

    print("Building the RepoLens AI keyword search index...")

    retriever = build_retriever()

    print(f"Indexed {len(retriever.chunks)} code chunks.")
    print("Ask a question about the repository.")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nYour question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting RepoLens AI.")
            break

        if not query:
            print("Please enter a question.")
            continue

        results = retriever.search(query, top_k=5)

        if not results:
            print("No relevant code chunks were found.")
            continue

        print(f"\nTop {len(results)} results:")

        for rank, result in enumerate(results, start=1):
            chunk = result.chunk

            print(f"\n[{rank}] Relevance score: {result.score:.4f}")
            print(f"File: {chunk.file_path}")
            print(
                f"Lines: {chunk.start_line}-{chunk.end_line}"
            )
            print(f"Symbol: {chunk.symbol_name}")
            print("-" * 60)

            preview = chunk.code[:700]
            print(preview)

            if len(chunk.code) > 700:
                print("\n... [preview truncated]")


if __name__ == "__main__":
    main()
