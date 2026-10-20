
import streamlit as st

import tempfile


import tempfile
import zipfile

from pathlib import Path

from src.upload_handler import extract_repository_zip
from src.repository_loader import find_python_files
from src.code_chunker import chunk_repository


from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.retriever import build_retriever
from src.vector_store import build_vector_store
from src.answer_generator import retrieve_evidence, generate_answer


from pathlib import Path

from src.upload_handler import extract_repository_zip
from src.repository_loader import find_python_files
from src.code_chunker import chunk_repository


load_dotenv()

GENERATION_MODEL = "gemini-3.8-flash"

st.set_page_config(
    page_title="RepoLens AI",
    page_icon="ðŸ”Ž",
    layout="wide",
)


st.markdown(
    """
    <style>
        .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
            padding-top: 2rem;
            max-width: 100%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


if "workspace_dir" not in st.session_state:
    st.session_state.workspace_dir = None

if "repository_root" not in st.session_state:
    st.session_state.repository_root = None

if "indexed_repository" not in st.session_state:
    st.session_state.indexed_repository = None


st.title("ðŸ”Ž RepoLens AI")
st.subheader("Understand any Python codebase with AI")

st.markdown(
    """
    Upload a Python repository as a ZIP file, ask questions
    about its implementation, and explore answers grounded
    in source code.
    """
)

st.info(
    "Your repository will be processed for code search. "
    "Do not upload secrets, API keys, or confidential code."
)

uploaded_file = st.file_uploader(
    "Upload your repository (.zip)",
    type=["zip"],
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Inspect repository"):
        new_workspace = None

        try:
            new_workspace = Path(
                tempfile.mkdtemp(prefix="repolens_")
            )

            zip_path = new_workspace / "repository.zip"
            extraction_path = new_workspace / "extracted"

            zip_path.write_bytes(uploaded_file.getvalue())

            repository_root = extract_repository_zip(
                zip_path,
                extraction_path,
            )

            python_files = find_python_files(repository_root)
            chunks = chunk_repository(repository_root)

            if not chunks:
                raise ValueError("No usable Python code chunks were found.")

            previous_workspace = st.session_state.workspace_dir

            st.session_state.workspace_dir = str(new_workspace)
            st.session_state.repository_root = str(repository_root)
            st.session_state.indexed_repository = None

            if previous_workspace:
                import shutil

                shutil.rmtree(previous_workspace, ignore_errors=True)

            st.success("Repository inspection completed!")
            st.write(f"Repository root: `{repository_root.name}`")
            st.write(f"Python files found: {len(python_files)}")
            st.write(f"Code chunks created: {len(chunks)}")

            with st.expander("View discovered Python files"):
                for file_path in python_files:
                    st.code(
                        file_path.relative_to(repository_root).as_posix(),
                        language="text",
                    )

        except (ValueError, OSError, zipfile.BadZipFile) as exc:
            if new_workspace:
                import shutil

                shutil.rmtree(new_workspace, ignore_errors=True)

            st.error(f"Could not inspect this repository: {exc}")

question = st.text_input(
    "Ask a question about your codebase",
    placeholder="How is the main model initialized?",
)


if st.button("Ask RepoLens AI", type="primary"):
    if uploaded_file is None:
        st.warning("Please upload a repository ZIP first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    elif not st.session_state.repository_root:
        st.warning("Please click 'Inspect repository' first.")
    else:
        try:
            repository_path = Path(
                st.session_state.repository_root
            )

            # Reuse the indexes for this repository during the session.
            if st.session_state.indexed_repository != str(repository_path):
                with st.spinner("Building keyword index..."):
                    bm25_retriever = build_retriever(repository_path)

                with st.spinner("Building semantic index with Gemini..."):
                    workspace = Path(st.session_state.workspace_dir)
                    vector_path = workspace / "chroma_db"

                    vector_store = build_vector_store(
                        repository_path=repository_path,
                        vector_store_path=vector_path,
                    )

                st.session_state.bm25_retriever = bm25_retriever
                st.session_state.vector_store = vector_store
                st.session_state.indexed_repository = str(repository_path)

            with st.spinner("Generating an answer from your code..."):
                llm = ChatGoogleGenerativeAI(
                    model=GENERATION_MODEL,
                    temperature=0,
                    timeout=30,
                    max_retries=1,
                )

                chunks = retrieve_evidence(
                    question,
                    st.session_state.bm25_retriever,
                    st.session_state.vector_store,
                    repository_path=repository_path,
                )

                answer = generate_answer(
                    question,
                    chunks,
                    llm,
                    repository_path=repository_path,
                )

            st.subheader("RepoLens AI Answer")
            st.markdown(answer)

            st.write(
                "DEBUG â€” retrieved files:",
                sorted({str(chunk.file_path) for chunk in chunks})
            )

            with st.expander("View retrieved source chunks"):
                for index, chunk in enumerate(chunks, start=1):
                    st.markdown(
                        f"**[{index}] {chunk.file_path}** "
                        f"(lines {chunk.start_line}â€“{chunk.end_line})"
                    )
                    st.code(chunk.code, language="python")





        except Exception as exc:

            error_message = str(exc).upper()

            quota_or_service_error = any(

                marker in error_message

                for marker in (

                    "429",

                    "RESOURCE_EXHAUSTED",

                    "503",

                    "UNAVAILABLE",

                )

            )

            if quota_or_service_error:

                st.warning(

                    "Gemini is currently unavailable or its quota "

                    "has been reached. Showing retrieved source "

                    "evidence instead of an AI-generated answer."

                )

                if "chunks" in locals() and chunks:

                    st.subheader("Retrieved source evidence")

                    for index, chunk in enumerate(chunks, start=1):
                        st.markdown(

                            f"**[{index}] {chunk.file_path}** "

                            f"(lines {chunk.start_line}â€“{chunk.end_line})"

                        )

                        st.code(chunk.code, language="python")

                else:

                    st.info(

                        "No source evidence is available yet. "

                        "Please try again."

                    )

            else:

                st.error(

                    f"Could not answer this question: "

                    f"{type(exc).__name__}: {exc}"

                )
