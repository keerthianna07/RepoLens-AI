
from src.retriever import build_retriever
from src.vector_store import (
    VECTOR_STORE_PATH,
    COLLECTION_NAME,
    GoogleGenerativeAIEmbeddings,
    Chroma,
    search_similar_code,
)
from src.hybrid_retriever import reciprocal_rank_fusion
from src.repository_inspector import find_model_architectures
from src.repository_loader import DEFAULT_REPOSITORY


TOP_K = 5

# Each question has a known relevant source file.
# These expected files let us measure retrieval quality.
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
    """Normalize paths so Windows and forward-slash paths can be compared."""
    return str(value).replace("\\", "/").lower()


def path_matches(actual_path, expected_path):
    """Check whether a retrieved path matches an expected source file."""
    actual = normalize_path(actual_path)
    expected = normalize_path(expected_path)

    return actual == expected or actual.endswith("/" + expected)


def evaluate_ranking(ranked_paths, expected_files):
    """
    Return:
    - hit: 1 if a relevant file appears in the ranking, otherwise 0
    - reciprocal rank: 1/rank of the first relevant result
    """
    for rank, actual_path in enumerate(ranked_paths, start=1):
        if any(
            path_matches(actual_path, expected)
            for expected in expected_files
        ):
            return 1, 1.0 / rank

    return 0, 0.0


def evaluate_retrieval():
    print("Initializing retrieval evaluation...")

    # Build the BM25 keyword index from the repository.
    bm25 = build_retriever()

    # Load the existing persistent vector database.
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

    methods = {
        "BM25": {"hits": 0, "rr_sum": 0.0},
        "Semantic": {"hits": 0, "rr_sum": 0.0},
        "Hybrid": {"hits": 0, "rr_sum": 0.0},
    }

    total = len(TEST_CASES)

    print(f"\nNumber of test questions: {total}")
    print(f"Evaluation cutoff: top {TOP_K}")

    for number, test in enumerate(TEST_CASES, start=1):
        question = test["question"]
        expected_files = test["expected_files"]

        print("\n" + "=" * 72)
        print(f"Test {number}: {question}")
        print(f"Expected file(s): {', '.join(expected_files)}")

        # Retrieve results using all three approaches.
        bm25_results = bm25.search(question, top_k=TOP_K)

        semantic_results = search_similar_code(
            vector_store,
            question,
            top_k=TOP_K,
        )

        hybrid_results = reciprocal_rank_fusion(
            bm25_results,
            semantic_results,
            top_k=TOP_K,
        )

        # Extract source paths in ranking order.
        ranked_paths = {
            "BM25": [
                str(result.chunk.file_path)
                for result in bm25_results
            ],
            "Semantic": [
                str(document.metadata["file_path"])
                for document, distance in semantic_results
            ],
            "Hybrid": [
                str(chunk.file_path)
                for chunk, score in hybrid_results
            ],
        }

        for method, paths in ranked_paths.items():
            hit, reciprocal_rank = evaluate_ranking(
                paths,
                expected_files,
            )

            methods[method]["hits"] += hit
            methods[method]["rr_sum"] += reciprocal_rank

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
                f"  {method:<8} | Hit@{TOP_K}: {hit} "
                f"| First relevant rank: {rank_text}"
            )

    # Calculate and display aggregate metrics.
    print("\n" + "=" * 72)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 72)
    print(f"{'Method':<12} {'Hit@5':>12} {'MRR@5':>12}")

    for method, scores in methods.items():
        hit_rate = scores["hits"] / total
        mrr = scores["rr_sum"] / total

        print(
            f"{method:<12} {hit_rate:>11.2%} {mrr:>12.4f}"
        )

    # Check whether static inspection recognizes the four known models.
    print("\nModel inventory coverage check:")

    discovered = find_model_architectures(DEFAULT_REPOSITORY)
    discovered_names = {
        item["model"] for item in discovered
    }

    expected_models = {
        "EfficientNetB0",
        "EfficientNetB3",
        "ResNet50",
        "VGG19",
    }

    found = expected_models & discovered_names
    missing = expected_models - discovered_names

    print(f"Expected models: {len(expected_models)}")
    print(f"Recognized models: {len(found)}")
    print(f"Inventory coverage: {len(found) / len(expected_models):.2%}")

    if missing:
        print(f"Not recognized: {', '.join(sorted(missing))}")

    print(
        "\nNote: These metrics measure file retrieval against this "
        "small test set. They do not establish answer correctness "
        "or prove that hybrid retrieval is better than either baseline."
    )


if __name__ == "__main__":
    evaluate_retrieval()
