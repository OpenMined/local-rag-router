import json
import uuid
from pathlib import Path
from typing import Any, Optional

import requests
from loguru import logger
from pydantic import BaseModel, Field
from syft_llm_router import BaseLLMRouter
from syft_llm_router.error import EmbeddingServiceError, EndpointNotImplementedError
from syft_llm_router.schema import (
    ChatResponse,
    CompletionResponse,
    DocumentResult,
    EmbeddingOptions,
    EmbeddingResponse,
    GenerationOptions,
    Message,
    RetrievalOptions,
    RetrievalResponse,
)
from tqdm import tqdm
from PyPDF2 import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import yaml


class Document(BaseModel):
    """Document structure from the JSON files"""

    # Unique identifier for the document
    id: str = Field(..., description="Unique identifier for the document")

    # Text content to be embedded
    content: str = Field(..., description="Content of the document")

    # Additional metadata associated with the document
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the document"
    )


class SyftRAGRouter(BaseLLMRouter):
    """Syft LLM Router class with RAG capabilities.

    This class is a placeholder for the actual implementation of the Syft LLM provider.
    It should be extended to implement the specific functionality required by your application.
    """

    def generate_chat(
        self,
        model: str,
        messages: list[Message],
        options: GenerationOptions | None = None,
    ) -> ChatResponse:
        return EndpointNotImplementedError("Generate chat method not implemented")

    def generate_completion(
        self, model: str, prompt: str, options: GenerationOptions | None = None
    ) -> CompletionResponse:
        return EndpointNotImplementedError("Generate completion method not implemented")

    def read_document(self, file_path: Path) -> list[dict] | None:
        """Read a document file (JSON, PDF, EPUB, Markdown, YAML) and return the contents as a list of dicts.

        Args:
            file_path: Path to the document file
        Returns:
            list[dict] | None: List of document dicts, or None if unsupported
        """
        ext = file_path.suffix.lower()
        if ext == ".json":
            return json.load(file_path.open("r"))
        elif ext == ".pdf":
            try:
                reader = PdfReader(str(file_path))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                return [{"content": text, "doc_id": str(uuid.uuid4())}]
            except Exception as e:
                logger.error(f"Error reading PDF file {file_path}: {e}")
                return None
        elif ext == ".epub":
            try:
                book = epub.read_epub(str(file_path))
                combined_text = ""
                
                # Function to extract text from HTML content
                def chapter_to_text(chapter_content):
                    soup = BeautifulSoup(chapter_content, 'html.parser')
                    text = soup.get_text()
                    # Clean up whitespace
                    return ' '.join(text.split())
                
                # Iterate through all items in the book
                for item in book.get_items():
                    # Check if the item is a document (HTML content)
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        # Get the content as bytes and decode to string
                        content = item.get_content().decode('utf-8')
                        # Extract text from the content
                        chapter_text = chapter_to_text(content)
                        # Add the chapter text to our combined text
                        combined_text += chapter_text + "\n\n"
                
                return [{"content": combined_text, "doc_id": str(uuid.uuid4())}]
            except Exception as e:
                logger.error(f"Error reading EPUB file {file_path}: {e}")
                return None
        elif ext == ".md":
            try:
                with open(file_path, 'r') as file:
                    text = file.read()
                return [{"content": text, "doc_id": str(uuid.uuid4())}]
            except Exception as e:
                logger.error(f"Error reading markdown file {file_path}: {e}")
                return None
        elif ext == ".yaml" or ext == ".yml":
            try:
                with open(file_path, 'r') as file:
                    text = file.read()
                return [{"content": text, "doc_id": str(uuid.uuid4())}]
            except Exception as e:
                logger.error(f"Error reading YAML file {file_path}: {e}")
                return None
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return None



    def embed_documents(
        self,
        watch_path: str,
        embedder_endpoint: str,
        indexer_endpoint: str,
        options: Optional[EmbeddingOptions] = None,
    ) -> EmbeddingResponse:
        """Process JSON files in a specified location and create embeddings for indexing.

        Args:
            watch_path: Directory path to watch for new JSON files
            embedder_endpoint: HTTP endpoint of the embedding service
            indexer_endpoint: HTTP endpoint of the indexing service
            options: Additional parameters for embedding configuration

        Returns:
            EmbeddingResponse: The result of the document embedding operation
        """

        logger.info(f"Processing documents from {watch_path}")

        file_path = Path(watch_path)

        total_documents = 0
        failed_documents = 0

        documents = self.read_document(file_path)

        if not documents:
            return EmbeddingResponse(
                id=uuid.uuid4(),
                status="success",
                processed_count=0,
                failed_count=0,
            )

        formatted_documents = self._format_raw_documents(documents)
        total_documents += len(formatted_documents)

        logger.info(f"Embedding {total_documents} documents")
        for document in tqdm(formatted_documents):
            if document.metadata.get("doc_id"):
                if self._is_document_present(
                    indexer_endpoint,
                    {"doc_id": document.metadata.get("doc_id")},
                ):
                    logger.info(f"Document {document.id} already exists in the index")
                    continue

            response = self._embed_document(embedder_endpoint, document)
            if "embeddings" not in response or not response["embeddings"]:
                failed_documents += 1
                logger.error(f"No embeddings found for document {document.id}")
                continue

            embeddings = response["embeddings"]

            try:
                logger.info(f"Indexing document {document.id}")
                self._index_document(indexer_endpoint, embeddings)
            except Exception as e:
                failed_documents += 1
                logger.error(f"Error indexing document {document.id}: {e}")
                continue

        logger.info(f"Total documents processed: {total_documents}")
        logger.info(f"Failed documents: {failed_documents}")

        # Unlink the file
        # TODO: Move to archive directory
        file_path.unlink()

        return EmbeddingResponse(
            id=uuid.uuid4(),
            status="success" if failed_documents == 0 else "partial_success",
            processed_count=total_documents,
            failed_count=failed_documents,
        )

    def _is_document_present(self, indexer_endpoint: str, filter: dict) -> bool:
        """Check if a document exists in the index.

        Args:
            indexer_endpoint: HTTP endpoint of the indexing service
            filter: Filter to check if a document exists
        """
        # Remove trailing slash if present
        indexer_endpoint = indexer_endpoint.rstrip("/")

        logger.info(f"Checking if document exists in the index with filter: {filter}")
        logger.info(f"Indexer endpoint: {indexer_endpoint}")

        response = requests.post(f"{indexer_endpoint}/count", json={"filter": filter})
        response.raise_for_status()

        if response.json()["count"] > 0:
            return True
        return False

    def _index_document(self, indexer_endpoint: str, embeddings: list[dict]) -> dict:
        """Index a document using the indexing service.

        Args:
            indexer_endpoint: HTTP endpoint of the indexing service
        """
        response = requests.post(
            f"{indexer_endpoint}/index", json={"embeddings": embeddings}
        )
        response.raise_for_status()
        return response.json()

    def _embed_document(
        self,
        embedder_endpoint: str,
        document: Document,
        options: Optional[EmbeddingOptions] = None,
    ) -> list[float]:
        """Embed a document using the embedding service.

        Args:
            document: Document to embed

        Returns:
            list[float]: The embedded document

        """

        # Based on what the embedding service supports,
        # create the request body

        optional_kwargs = (
            {
                "chunk_size": options.chunk_size,
                "chunk_overlap": options.chunk_overlap,
                "batch_size": options.batch_size,
            }
            if options
            else {}
        )

        request_body = {
            "document": document.model_dump(mode="json"),
            **optional_kwargs,
        }

        response = requests.post(f"{embedder_endpoint}/embed", json=request_body)
        response.raise_for_status()
        response_json = response.json()
        return response_json

    def _format_raw_documents(self, raw_documents: list[dict]) -> list[Document]:
        """Format the documents to the Document structure.

        Args:
            document: Dictionary representation of the document

        Returns:
            Document: The formatted document
        """
        formatted_documents = []
        for raw_document in raw_documents:
            document_dict = {
                "id": str(uuid.uuid4()),
                "content": raw_document.get("content"),
                "metadata": {
                    "doc_id": raw_document.get("doc_id"),
                    "top_image": raw_document.get("top_image"),
                    "title": raw_document.get("title"),
                    "link": raw_document.get("link"),
                },
            }
            document = Document(**document_dict)
            formatted_documents.append(document)

        return formatted_documents

    def retrieve_documents(
        self,
        query: str,
        embedder_endpoint: str,
        retriever_endpoint: str,
        options: Optional[RetrievalOptions] = None,
    ) -> RetrievalResponse:
        """Retrieve documents from the index based on a search query.

        Args:
            query: Search query to find relevant documents
            embedder_endpoint: HTTP endpoint of the embedding service
            retriever_endpoint: HTTP endpoint of the retriever service
            options: Additional parameters for retrieval configuration

        Returns:
            RetrievalResponse: The result of the document retrieval operation
        """
        logger.info(f"Retrieving documents for query: {query}")

        document_id = str(uuid.uuid4())
        document = Document(
            id=document_id,
            content=query,
            metadata={"type": "query"},
        )
        response = self._embed_document(embedder_endpoint, document)
        if "embeddings" not in response or not response["embeddings"]:
            raise EmbeddingServiceError(f"No embeddings found for query {query}")

        retriever_request = {
            "query": response["embeddings"][0]["vector"],
            "top_k": options.limit if options else 3,
        }

        logger.info(f"Retriever request: {retriever_request}")
        endpoint = retriever_endpoint.rstrip("/")
        retriever_route = endpoint + "/search"

        response = requests.post(retriever_route, json=retriever_request)
        response.raise_for_status()
        response_json = response.json()

        document_results = []

        for result in response_json["results"]:
            document_results.append(
                DocumentResult(
                    id=result["document_id"],
                    score=result["score"],
                    content=result["content"],
                    metadata=result["metadata"],
                )
            )

        return RetrievalResponse(
            id=uuid.uuid4(),
            query=query,
            results=document_results,
        )


if __name__ == "__main__":
    router = SyftRAGRouter()
    # # response = router.embed_documents(
    # #     "/home/shubham/repos/OpenMined/syft-rag/data/data.json",
    # #     "http://localhost:8002",
    # #     "http://localhost:8001",
    # # )
    # # print(response)

    # response = router.retrieve_documents(
    #     "What is the use of AI?",
    #     "http://localhost:8002",
    #     "http://localhost:8001",
    # )
    # print(response)
