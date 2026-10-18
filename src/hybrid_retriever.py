
from pathlib import Path

from src.retriever import BM25Retriever, build_retriever
from src.vector_store import (
    VECTOR_STORE_PATH,
    COLLECTION_NAME,
    GoogleGenerativeAIEmbeddings,
    Chroma,
    search_similar_code,
)


RRF_CONSTANT = 60


def reciprocal_rank_fusion(
    bm25_results,
    vector_results,
    top_k: int = 5,
):
    """Combine BM25 and semantic rankings using reciprocal rank fusion."""

    if top_k < 1:
        return []

    fused_scores = {}
    result_chunks = {}

    # BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        chunk = result.chunk
        key = (
            chunk.file_path,
            chunk.start_line,
            chunk.end_line,
            chunk.symbol_name,
        )

        fused_scores[key] = fused_scores.get(key, 0.0) + (
            1.0 / (RRF_CONSTANT + rank)
        )
        result_chunks[key] = chunk

    # Semantic results
    for rank, (document, distance) in enumerate(
        vector_results,
        start=1,
    ):
        metadata = document.metadata
        key = (
            metadata["file_path"],
            metadata["start_line"],
            metadata["end_line"],
            metadata["symbol_name"],
        )

        fused_scores[key] = fused_scores.get(key, 0.0) + (
            1.0 / (RRF_CONSTANT + rank)
        )

        # Reuse the original CodeChunk when available.
        if key not in result_chunks:
            from src.code_chunker import CodeChunk

            result_chunks[key] = CodeChunk(
                file_path=Path(metadata["file_path"]),
                symbol_name=metadata["symbol_name"],
                start_line=int(metadata["start_line"]),
                end_line=int(metadata["end_line"]),
                code=document.page_content,
                chunk_type=metadata["chunk_type"],
            )

    ranked_keys = sorted(
        fused_scores,
        key=lambda key: fused_scores[key],
        reverse=True,
    )

    return [
        (result_chunks[key], fused_scores[key])
        for key in ranked_keys[:top_k]
    ]


def main() -> None:
    """Run a question through both retrievers and fuse their rankings."""

    print("Building BM25 retriever...")
    bm25_retriever: BM25Retriever = build_retriever()

    print("Connecting to the existing ChromaDB index...")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_STORE_PATH),
    )

    print("Hybrid retrieval is ready.")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nYour question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting RepoLens AI.")
            break

        if not query:
            print("Please enter a question.")
            continue

        bm25_results = bm25_retriever.search(query, top_k=5)

        vector_results = search_similar_code(
            vector_store,
            query,
            top_k=5,
        )

        hybrid_results = reciprocal_rank_fusion(
            bm25_results,
            vector_results,
            top_k=5,
        )

        print("\nTop hybrid retrieval results:")

        if not hybrid_results:
            print("No relevant code chunks found.")
            continue

        for rank, (chunk, score) in enumerate(
            hybrid_results,
            start=1,
        ):
            print(f"\n[{rank}] RRF score: {score:.6f}")
            print(f"File: {chunk.file_path}")
            print(f"Lines: {chunk.start_line}-{chunk.end_line}")
            print(f"Symbol: {chunk.symbol_name}")
            print("-" * 60)
            print(chunk.code[:700])

            if len(chunk.code) > 700:
                print("\n... [preview truncated]")


if __name__ == "__main__":
    main()
