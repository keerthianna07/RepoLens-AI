from src.repository_inspector import find_model_architectures
from src.repository_loader import DEFAULT_REPOSITORY

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.hybrid_retriever import (
    reciprocal_rank_fusion,
)
from src.retriever import build_retriever
from src.vector_store import (
    VECTOR_STORE_PATH,
    COLLECTION_NAME,
    GoogleGenerativeAIEmbeddings,
    Chroma,
    search_similar_code,
)


load_dotenv()

GENERATION_MODEL = "gemini-3.8-flash"
RETRIEVAL_COUNT = 8


def retrieve_evidence(
    question: str,
    bm25_retriever,
    vector_store: Chroma,
    top_k: int = RETRIEVAL_COUNT,
) -> list:
    """Retrieve and combine relevant code chunks."""

    keyword_results = bm25_retriever.search(
        question,
        top_k=top_k,
    )

    semantic_results = search_similar_code(
        vector_store,
        question,
        top_k=top_k,
    )

    fused_results = reciprocal_rank_fusion(
        keyword_results,
        semantic_results,
        top_k=top_k,
    )

    return [chunk for chunk, score in fused_results]


def build_evidence_context(chunks: list) -> str:
    """Format retrieved chunks with traceable source references."""

    if not chunks:
        return ""

    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        source_reference = (
            f"[S{index}] {chunk.file_path}:"
            f"L{chunk.start_line}-L{chunk.end_line}"
        )

        context_parts.append(
            f"{source_reference}\n"
            f"Symbol: {chunk.symbol_name}\n"
            f"Code:\n{chunk.code}"
        )

    return "\n\n" + ("\n\n" + "=" * 70 + "\n\n").join(
        context_parts
    )

def is_model_inventory_question(question: str) -> bool:
    """Identify questions asking for a repository-wide model inventory."""

    question_lower = question.lower()

    inventory_phrases = (
        "which models",
        "what models",
        "which deep learning models",
        "what deep learning models",
        "models are implemented",
        "models are present",
        "model architectures",
        "list the models",
        "list all models",
    )

    return any(
        phrase in question_lower
        for phrase in inventory_phrases
    )


def build_model_inventory_context() -> str:
    """Build source-grounded evidence from the AST inspector."""

    models = find_model_architectures(DEFAULT_REPOSITORY)

    if not models:
        return "No recognized model architecture references were found."

    lines = [
        "Repository-wide AST inspection results:",
        "These are recognized architecture references in source code.",
        "They do not prove that model training completed successfully.",
    ]

    for index, model in enumerate(models, start=1):
        lines.append(
            f"[I{index}] {model['model']} — "
            f"{model['file']}, line {model['line']}"
        )

    return "\n".join(lines)

def generate_answer(
    question: str,
    chunks: list,
    llm: ChatGoogleGenerativeAI,
) -> str:
    """Generate an explanation grounded in retrieved source code."""

    if not question.strip():
        return "Please enter a question."

    if not chunks:
        return (
            "I could not find relevant source code to answer "
            "this question. Try asking about a specific model, "
            "function, file, or processing step."
        )
    context = build_evidence_context(chunks)

    inventory_context = ""

    if is_model_inventory_question(question):
        inventory_context = build_model_inventory_context()

    prompt = f"""
You are RepoLens AI, a source-code understanding assistant.

Your task is to answer questions about a Python repository
using ONLY the source-code evidence supplied below.

USER QUESTION:
{question}

RETRIEVED SOURCE CODE:
{context}

REPOSITORY-WIDE MODEL INVENTORY:
{inventory_context}

RULES:
1. Answer the user's actual question directly.
2. Do not invent functions, files, model architectures,
   parameters, results, or implementation details.
3. Cite supporting evidence using the exact source IDs and
   file/line references supplied above.
4. Use citations in this format:
   [S1: src/example.py, lines 10-20]
5. Only cite a source when its code supports the associated claim.
6. If the evidence is incomplete, explicitly state what cannot
   be verified from the retrieved chunks.
7. If the supplied evidence is insufficient to answer, say so
   instead of guessing.
8. Distinguish code that is defined from code that is actually
   executed. Do not claim that a model was trained successfully
   merely because a training call exists in the source.
9. Use clear, concise language. Explain technical terms when
   helpful.
10. Do not claim to have inspected source files that are absent
    from the supplied evidence.
11. When repository-wide inventory evidence is supplied, use it
    to answer inventory questions rather than relying only on
    the retrieved chunks.
12. Cite inventory entries using:
    [I1: src/train_vgg19.py, line 41]
13. Describe these as architecture references found in source
    code. Do not claim successful training or execution unless
    the evidence establishes it.

Write the answer now.
"""

    response = llm.invoke(prompt)
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                text_parts.append(item.text)

        if text_parts:
            return "\n".join(text_parts)

    return str(content)


def main() -> None:
    """Run the complete retrieval and answer-generation pipeline."""

    print("Initializing RepoLens AI...")

    bm25_retriever = build_retriever()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_STORE_PATH),
    )

    llm = ChatGoogleGenerativeAI(
        model=GENERATION_MODEL,
        temperature=0,
    )

    print("RepoLens AI is ready.")
    print("Ask a question about the repository.")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Exiting RepoLens AI.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            chunks = retrieve_evidence(
                question,
                bm25_retriever,
                vector_store,
            )

            answer = generate_answer(
                question,
                chunks,
                llm,
            )

            print("\nRepoLens AI Answer")
            print("=" * 70)
            print(answer)

        except Exception as error:
            print("\nAn error occurred while processing your question.")
            print(f"Error type: {type(error).__name__}")
            print(f"Details: {error}")


if __name__ == "__main__":
    main()
