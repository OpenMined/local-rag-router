import argparse
from pathlib import Path
from typing import Optional, Union
import uuid

from loguru import logger
from pydantic import BaseModel
from syft_core import Client
from syft_event import SyftEvents
from syft_event.types import Request
from syft_llm_router import BaseLLMRouter
from syft_llm_router.error import (
    EndpointNotImplementedError,
    Error,
    InvalidRequestError,
)
from syft_llm_router.schema import (
    ChatResponse,
    CompletionResponse,
    EmbeddingOptions,
    GenerationOptions,
    Message,
    RetrievalOptions,
    RetrievalResponse,
)
from router import SyftRAGRouter, DocumentResult
import requests
# from watchdog.events import FileSystemEvent

# Global variable to store the dynamic port
PORT = None  



class ChatRequest(BaseModel):
    """Chat request model."""

    model: str
    messages: list[Message]
    options: Optional[GenerationOptions] = None


class DocumentRetrievalRequest(BaseModel):
    """Document retrieval request model."""

    query: str
    options: Optional[RetrievalOptions] = None


class CompletionRequest(BaseModel):
    """Completion request model."""

    model: str
    prompt: str
    options: Optional[GenerationOptions] = None


def load_router() -> BaseLLMRouter:
    """Load your implementation of the LLM provider."""

    # This is a placeholder for the actual provider loading logic.
    # You should replace this with the actual provider you want to use.
    # For example, if you have a provider class named `MyLLMProvider`, you would do:
    # from my_llm_provider import MyLLMProvider
    # args = ...  # Load or define your provider arguments here
    # kwargs = ...  # Load or define your provider keyword arguments here
    # provider = MyLLMProvider(*args, **kwargs)
    # return provider

    from router import SyftRAGRouter

    return SyftRAGRouter()


def get_embedder_endpoint() -> str:
    """Get the embedder endpoint."""
    return "http://localhost:8002"


def get_indexer_endpoint() -> str:
    """Get the indexer endpoint."""
    return "http://localhost:8001"


def get_retriever_endpoint() -> str:
    """Get the retriever endpoint."""
    return "http://localhost:8001"


def create_server(project_name: str, config_path: Optional[Path] = None):
    """Create and return the SyftEvents server with the given config path."""
    if config_path:
        client = Client.load(path=config_path)
    else:
        client = Client.load()

    server_name = f"llm/{project_name}"
    return SyftEvents(server_name, client=client)

def get_port(log_file_path):
    import re
    with open(log_file_path, 'r') as file:
        for line in file:
            if "[syftbox] App Port:" in line:
                port_match = re.search(r'App Port:\s*(\d+)', line)
                if port_match:
                    return int(port_match.group(1))
    return None

def handle_completion_request(
    request: CompletionRequest,
    ctx: Request,
) -> Union[CompletionResponse, Error]:
    return EndpointNotImplementedError(message="Not implemented")


def handle_chat_completion_request(
    request: ChatRequest,
    ctx: Request,
) -> Union[ChatResponse, Error]:
    return EndpointNotImplementedError(message="Not implemented")
    

def handle_document_retrieval_request(
    request: DocumentRetrievalRequest,
    ctx: Request,
) -> Union[RetrievalResponse, Error]:
    """Handle a document retrieval request."""
    logger.info(f"Processing document retrieval request: <{ctx.id}> from <{ctx.sender}>")
    logger.info(f"Request query: {request.query}")
    logger.info(f"Request options: {request.options}")
   
    try:
        # Make POST request to local retrieval endpoint
        if PORT is None:
            logger.error("Retrieval port is not set. Aborting request.")
            raise Exception("Retrieval port is not set.")
        retrieval_url = f"http://localhost:{PORT}/api/search-paths"
        logger.info(f"Retrieval URL: {retrieval_url}")
        payload = {
            "query": request.query,
            "limit": request.options.limit if request.options and hasattr(request.options, 'limit') else 5
        }
        logger.info(f"Payload for retrieval: {payload}")
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            retrieval_url,
            json=payload, 
            headers=headers
        )
        logger.info(f"POST request sent. Status code: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Retrieval request failed with status {response.status_code} and response: {response.text}")
            raise Exception(f"Retrieval request failed with status {response.status_code}")
            
        # Convert response to RetrievalResponse format
        results = response.json()
        logger.info(f"Raw response JSON: {results}")
        file_paths = results.get("file_paths", [])
        document_results = [
            DocumentResult(
                id=path,  # Use the file path as the id
                score=1.0,  # Default score
                content="",  # No file content included
                metadata={},
                embedding=None
            ) for path in file_paths
        ]
        logger.info(f"Parsed document results: {document_results}")
        
        response = RetrievalResponse(
            id=uuid.uuid4(),
            query=request.query,
            results=document_results,
        )
        logger.info(f"Returning RetrievalResponse: {response}")
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        response = InvalidRequestError(message=str(e))
    return response


# def handle_document_embeddings(event: FileSystemEvent) -> Optional[Error]:
#     """Handle a document embeddings request."""
#     logger.info(f"Listening for changes to {event.src_path}")
#     provider = load_router()
#     embedder_endpoint = get_embedder_endpoint()
#     indexer_endpoint = get_indexer_endpoint()

#     options = EmbeddingOptions(
#         chunk_size=1024,
#         chunk_overlap=2048,
#         batch_size=10,
#         process_interval=10,
#     )

#     try:
#         response = provider.embed_documents(
#             watch_path=event.src_path,
#             embedder_endpoint=embedder_endpoint,
#             indexer_endpoint=indexer_endpoint,
#             options=options,
#         )
#     except Exception as e:
#         logger.error(f"Error processing request: {e}")
#         response = InvalidRequestError(message=str(e))

#     return response


def ping(ctx: Request) -> str:
    """Ping the server."""
    return "pong"


def create_embedding_directory(datasite_path: Path):
    """Create a directory for embeddings."""
    # Create the embeddings directory
    embeddings_dir = datasite_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created embeddings directory at {embeddings_dir}")
    return embeddings_dir


def register_routes(server):
    """Register all routes on the given server instance."""
    server.on_request("/completions")(handle_completion_request)
    server.on_request("/chat")(handle_chat_completion_request)
    server.on_request("/retrieve")(handle_document_retrieval_request)
    server.on_request("/ping")(ping)

    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Syft LLM Router server")

    parser.add_argument(
        "--project-name",
        type=str,
        help="Name of the project instance.",
        required=True,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to client configuration file",
        required=False,
    )

    args = parser.parse_args()

    # Create server with config path
    box = create_server(project_name=args.project_name, config_path=args.config)

    LOG_FILE_PATH = box.client.config.data_dir / "apps/com.github.snwagh.syftbox-rag/logs/app.log"
    if not LOG_FILE_PATH.exists():
        print("Vectorize app not installed, please install it first")
        exit(1)
        
    port = get_port(LOG_FILE_PATH)
    print(f"Vectorize detected at port: {port}")
    PORT = port

    # Register routes
    register_routes(box)
    
    # Create the embeddings directory
    # create_embedding_directory(box.client.my_datasite)

    try:
        print("Starting server...")
        box.run_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error running server: {e}")
