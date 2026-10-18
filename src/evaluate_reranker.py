
"""Compare baseline retrieval methods with cross-encoder reranking."""

import time

from src.retriever import build_retriever
from src.vector_store import (
    VECTOR_STORE_PATH,
    COLLECTION_NAME,
    GoogleGenerativeAIEmbeddings,
    Chroma,
    search_similar_code,
)
from src.hybrid_retriever import reciprocal_rank_fusion
from src.reranker import CodeReranker


TOP_K = 5
CANDIDATE_K = 10

TEST_CASES = [
    {
        "question": "How is the EfficientNetB3 architecture defined?",
        "expected_files": ["src/efficientnet_b3.py"],
    },
    {
        "question": "How does the ResNet50 script evaluate predictions?",
        "expected_files": ["src/train_resnet50.py"],
    },
    {
        "question": "How is the VGG19 classifier trained?",
        "expected_files": ["src/train_vgg19.py"],
    },
    {
        "question": "Where is EfficientNetB0 fine-tuned?",
        "expected_files": ["src/efficientnet_b0.py"],
    },
    {
        "question": "Where are CLAHE enhanced images created?",
        "expected_files": ["src/clahe_preprocessing.py"],
    },
    {
        "question": "Which model uses an input shape of 300 by 300?",
        "expected_files": ["src/efficientnet_b3.py"],
    },
    {
        "question": "Where is ResNet50 initialized?",
        "expected_files": ["src/train_resnet50.py"],
    },
    {
        "question": "Where are class weights calculated?",
        "expected_files": ["src/efficientnet_b0.py"],
    },
]


def normalize_path(value):
    """Normalize Windows and forward-slash paths."""
    return str(value).replace("\\", "/").lower()


def path_matches(actual_path, expected_path):
    """Check whether a retrieved path matches an expected file."""
    actual = normalize_path(actual_path)
    expected = normalize_path(expected_path)

    return actual == expected or actual.endswith("/" + expected)


def evaluate_ranking(ranked_paths, expected_files):
    """Return Hit@K and reciprocal rank for one question."""
    for rank, actual_path in enumerate(ranked_paths, start=1):
        if any(
            path_matches(actual_path, expected)
            for expected in expected_files
        ):
            return 1, 1.0 / rank

    return 0, 0.0


def evaluate_reranker():
    print("Initializing comparative evaluation...")

    # Build the keyword retriever.
    bm25 = build_retriever()

    # Load the existing vector database without rebuilding it.
    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(
            "ChromaDB index not found. Run "
            "'python -m src.vector_store' first."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_STORE_PATH),
    )

    # Load the cross-encoder only once, not once per question.
    reranker = CodeReranker()

    method_names = [
        "BM25",
        "Semantic",
        "Hybrid",
        "Hybrid+Reranker",
    ]

    metrics = {
        name: {"hits": 0, "rr_sum": 0.0}
        for name in method_names
    }

    rerank_times = []
    total_questions = len(TEST_CASES)

    print(f"\nTest questions: {total_questions}")
    print(f"Final ranking cutoff: top {TOP_K}")
    print(f"Hybrid candidate pool: top {CANDIDATE_K}")

    for number, test in enumerate(TEST_CASES, start=1):
        question = test["question"]
        expected_files = test["expected_files"]

        print("\n" + "=" * 72)
        print(f"Test {number}: {question}")
        print(f"Expected: {', '.join(expected_files)}")

        # Retrieve candidates from both existing baselines.
        bm25_results = bm25.search(
            question,
            top_k=CANDIDATE_K,
        )

        semantic_results = search_similar_code(
            vector_store,
            question,
            top_k=CANDIDATE_K,
        )

        # Fuse keyword and semantic rankings.
        hybrid_candidates = reciprocal_rank_fusion(
            bm25_results,
            semantic_results,
            top_k=CANDIDATE_K,
        )

        # Reorder the hybrid candidates with the cross-encoder.
        start_time = time.perf_counter()

        reranked_results = reranker.rerank(
            question,
            hybrid_candidates,
            top_k=TOP_K,
        )

        elapsed = time.perf_counter() - start_time
        rerank_times.append(elapsed)

        # Compare the top five results for every method.
        ranked_paths = {
            "BM25": [
                str(result.chunk.file_path)
                for result in bm25_results[:TOP_K]
            ],
            "Semantic": [
                str(document.metadata["file_path"])
                for document, distance in semantic_results[:TOP_K]
            ],
            "Hybrid": [
                str(chunk.file_path)
                for chunk, score in hybrid_candidates[:TOP_K]
            ],
            "Hybrid+Reranker": [
                str(chunk.file_path)
                for chunk, score in reranked_results
            ],
        }

        for method, paths in ranked_paths.items():
            hit, reciprocal_rank = evaluate_ranking(
                paths,
                expected_files,
            )

            metrics[method]["hits"] += hit
            metrics[method]["rr_sum"] += reciprocal_rank

            first_match = next(
                (
                    rank
                    for rank, path in enumerate(paths, start=1)
                    if any(
                        path_matches(path, expected)
                        for expected in expected_files
                    )
                ),
                None,
            )

            rank_text = (
                str(first_match)
                if first_match is not None
                else f"Not in top {TOP_K}"
            )

            print(
                f"{method:<18} | "
                f"Hit@{TOP_K}: {hit} | "
                f"First relevant rank: {rank_text}"
            )

        print(f"Cross-encoder reranking time: {elapsed:.3f} seconds")

    # Aggregate results over the entire test set.
    print("\n" + "=" * 72)
    print("FINAL COMPARISON")
    print("=" * 72)
    print(f"{'Method':<20} {'Hit@5':>12} {'MRR@5':>12}")

    for method in method_names:
        scores = metrics[method]
        hit_rate = scores["hits"] / total_questions
        mrr = scores["rr_sum"] / total_questions

        print(
            f"{method:<20} {hit_rate:>11.2%} {mrr:>12.4f}"
        )

    average_latency = sum(rerank_times) / len(rerank_times)

    print("\nReranking latency")
    print(f"Total reranking time: {sum(rerank_times):.3f} seconds")
    print(f"Average per question: {average_latency:.3f} seconds")

    baseline_mrr = (
        metrics["Hybrid"]["rr_sum"] / total_questions
    )
    reranked_mrr = (
        metrics["Hybrid+Reranker"]["rr_sum"] / total_questions
    )

    print("\nCross-encoder comparison")
    print(f"Hybrid MRR@5: {baseline_mrr:.4f}")
    print(f"Reranked MRR@5: {reranked_mrr:.4f}")
    print(f"MRR change: {reranked_mrr - baseline_mrr:+.4f}")

    print(
        "\nNote: These are preliminary results on a small, "
        "repository-specific test set. They measure file retrieval, "
        "not generated-answer correctness. Reranking latency excludes "
        "initial model loading and the earlier retrieval calls."
    )


if __name__ == "__main__":
    evaluate_reranker()
