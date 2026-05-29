import sys
import os
from typing import Dict, Any

from chromadb.api.models.Collection import Collection
from loguru import logger
from chromadb import PersistentClient
import ollama
import re
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from backend.fastapi.langgraph.helpers.graph_state_classes import BusinessState

OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
)


def sanitize_chroma_collection_name(name: str) -> str:
    """
    Convert a string into a valid Chroma collection name.

    Rules:
    - Only a-z, A-Z, 0-9, `.`, `_`, `-`
    - Must start and end with a-z, A-Z, or 0-9
    - Length must be 3 to 512 characters
    """
    # Convert to lowercase
    name = name.lower()

    # Replace spaces and disallowed characters with "_"
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)

    # Remove leading/trailing non-alphanumeric characters
    name = re.sub(r"^[^a-zA-Z0-9]+", "", name)
    name = re.sub(r"[^a-zA-Z0-9]+$", "", name)

    # Ensure minimum and maximum length
    if len(name) < 3:
        name = name.ljust(3, "_")
    elif len(name) > 512:
        name = name[:512]

    return name


def flatten_business_state(state: BusinessState) -> str | None:
    """Transforms BusinessState into a string for vector storing
    :param state: BusinessState - pass in the business object
    :returns str | None - return the string if Business State can be converted, else return none
    """
    try:
        str_state_flat_parts = [
            f"Business Name: {state['business_name']}",
            f"Description: {state['business_description']}",
            f"Activity: {state['business_activity']}",
            f"Location: {state['business_location']}",
        ]
        for asset in state["assets"]["assets"]:
            str_state_flat_parts.append(
                f"Asset Category: {asset['category']} - {asset['description']}"
            )

        return "\n".join(str_state_flat_parts)

    except Exception as e:
        logger.error(e)
        return None


def load_markdown(file_path: str) -> str:
    """Load markdown content from file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()
            return markdown_text
    except FileNotFoundError:
        logger.error("File not found!")


def custom_numbered_header_split(markdown_text: str) -> list:
    """Split markdown by numbered headers and extract metadata."""
    pattern = r"(?=^\d+\.\s+#+\s+.*$)"
    raw_chunks = re.split(pattern, markdown_text, flags=re.MULTILINE)

    docs = []
    for chunk in raw_chunks:
        cleaned = chunk.strip()
        if not cleaned:
            continue

        lines = cleaned.splitlines()
        header_line = lines[0] if lines else ""

        # Extract header info
        match = re.match(r"^(\d+)\.\s+(#+)\s+(.*)", header_line)
        if match:
            number = match.group(1)
            level = len(match.group(2))
            title = match.group(3).strip()
            metadata = {"section_number": number, "level": level, "title": title}
        else:
            metadata = {}

        docs.append(Document(page_content=cleaned, metadata=metadata))

    return docs


def embed_text(text_to_embed: str, embedding_function: str = "mxbai-embed-large"):
    """Takes a flat string and embeds it. This function can be used to embed user query and the business info
    :param text_to_embed:str - the flattened business state string
    :param embedding_function:str - the name of the embedding function to use from ollama
    :param reason:str - the reason for the embedding. This is more of a logging parameter, leave as is
    """
    try:
        #response = ollama.embed(model=f"{embedding_function}", input=f"{text_to_embed}")
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.embed(model=f"{embedding_function}", input=f"{text_to_embed}")

        return response["embeddings"][0]

    except Exception as e:
        logger.error(e)
        return None


def update_chroma_collection(
    collection: Collection,
    collection_name: str,
    documents: list,
    embeddings: list,
    ids: list,
):
    """Updates the existing collection in the current Chroma database"""
    try:
        collection.add(documents=documents, embeddings=embeddings, ids=ids)
        logger.info(f"Updated collection: {collection_name}")
        return None
    except Exception as e:
        logger.error(e)
        return None


def ingest_business_profile(
    business_state: BusinessState,
    db_path: str = "../chromadb_vectorstore",
    embedding_model: str = "mxbai-embed-large",
):
    """Brings together the helper functions to ingest the business into a vectorstore"""
    collection_name = sanitize_chroma_collection_name(business_state["business_name"])
    flat_business = flatten_business_state(business_state)
    if not flat_business:
        logger.warning("Failed to flatten business state.")
        return

    embedding = embed_text(flat_business, embedding_function=embedding_model)
    if not embedding:
        logger.warning("Failed to embed business.")
        return

    # Create or fetch collection
    client = PersistentClient(db_path)
    existing_collections = [col.name for col in client.list_collections()]
    if collection_name in existing_collections:
        collection = client.get_collection(collection_name)
    else:
        collection_name = sanitize_chroma_collection_name(collection_name)
        collection = client.create_collection(collection_name)

    doc_id = business_state["business_name"].lower().replace(" ", "_")

    update_chroma_collection(
        collection=collection,
        collection_name=collection_name,
        documents=[flat_business],
        embeddings=[embedding],
        ids=[doc_id],
    )


def get_vectorstore(
    collection_name: str,
    #db_path: str = "../chromadb_vectorstore",
    db_path: str = "backend/chromadb_vectorstore",
    embedding_model: str = "mxbai-embed-large",
):
    """
    Returns a Chroma vectorstore for the specified collection.

    :param collection_name: The name of the ChromaDB collection
    :param db_path: Path to the ChromaDB directory
    :param embedding_model: Ollama model name for embeddings
    :return: Chroma vectorstore instance
    """
    try:
        sanitized_name = sanitize_chroma_collection_name(collection_name)

        # Set up LangChain-compatible embedding function
        #embedding_function: Embeddings = OllamaEmbeddings(model=embedding_model)
        embedding_function: Embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=OLLAMA_BASE_URL,
        )


        vectorstore = Chroma(
            client=PersistentClient(path=db_path),
            collection_name=sanitized_name,
            embedding_function=embedding_function,
        )

        return vectorstore

    except Exception as e:
        logger.error(f"Failed to get vectorstore: {e}")
        raise


def setup_vectorstore_saa(
    file_name: str,
    persist_dir: str = "../chromadb_vectorstore",
    embedding_model: str = "mxbai-embed-large",
    input_file_path: str = "../input_files/Security Assessment",
):
    """Initialize vector store. Load vectorstore if exists."""
    """
    collection_name = sanitize_chroma_collection_name(file_name)
    vectorstore_path = os.path.join(persist_dir, collection_name)

    if os.path.exists(vectorstore_path) and os.listdir(vectorstore_path):
        logger.info(f"Loading existing vectorstore from {vectorstore_path}")
        vectorstore = get_vectorstore(collection_name=collection_name)
    else:
        logger.info(f"Creating new vectorstore at {vectorstore_path}")
        # Create new vectorstore
        vectorstore = Chroma.from_documents(
            documents=custom_numbered_header_split(load_markdown(input_file_path)),
            embedding=OllamaEmbeddings(
                model=embedding_model,
                base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            ),
            persist_directory=persist_dir,
            collection_name=collection_name,
        )

    return vectorstore
    """
    collection_name = sanitize_chroma_collection_name(file_name)
    split_docs = custom_numbered_header_split(load_markdown(input_file_path))

    client = PersistentClient(path=persist_dir)
    existing_collections = [col.name for col in client.list_collections()]

    if collection_name in existing_collections:
        logger.info(f"Loading existing vectorstore collection: {collection_name}")
        vectorstore = get_vectorstore(
            collection_name=collection_name,
            db_path=persist_dir,
            embedding_model=embedding_model,
        )
    else:
        logger.info(f"Creating new vectorstore collection: {collection_name}")
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=OllamaEmbeddings(
                model=embedding_model,
                base_url=OLLAMA_BASE_URL,
            ),
            persist_directory=persist_dir,
            collection_name=collection_name,
        )

    return vectorstore, split_docs
