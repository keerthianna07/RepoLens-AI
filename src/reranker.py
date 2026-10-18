
"""Cross-encoder reranking for RepoLens AI."""

from sentence_transformers import CrossEncoder

from src.code_chunker import CodeChunk


# A lightweight cross-encoder pretrained for passage relevance.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CodeReranker:
    """Reorder retrieved code chunks according to query relevance."""

    def __init__(self, model_name: str = RERANKER_MODEL):
        print(f"Loading cross-encoder model: {model_name}")

        self.model = CrossEncoder(
            model_name,
            max_length=512,
            device="cpu",
        )

        print("Cross-encoder loaded successfully.")

    @staticmethod
    def _chunk_text(chunk: CodeChunk) -> str:
        """Build a text representation of a code chunk."""

        return (
            f"File: {chunk.file_path}\n"
            f"Symbol: {chunk.symbol_name}\n"
            f"Chunk type: {chunk.chunk_type}\n"
            f"Code:\n{chunk.code}"
        )

    def rerank(
        self,
        question: str,
        candidates: list[tuple[CodeChunk, float]],
        top_k: int = 5,
    ) -> list[tuple[CodeChunk, float]]:
        """
        Rerank retrieved chunks.

        Input scores are not used to determine the final order.
        The cross-encoder independently scores each question-chunk pair.

        Returns:
            List of (CodeChunk, cross_encoder_score), highest score first.
        """

        if not candidates:
            return []

        if top_k <= 0:
            return []

        pairs = [
            (question, self._chunk_text(chunk))
            for chunk, _original_score in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
            batch_size=8,
        )

        reranked = [
            (chunk, float(score))
            for (chunk, _original_score), score
            in zip(candidates, scores)
        ]

        reranked.sort(key=lambda item: item[1], reverse=True)

        return reranked[:top_k]
