from loguru import logger
from syft_core import Client
from syft_llm_router.error import RouterError
from syft_llm_router.schema import RetrievalResponse
from syft_rpc import rpc

APP_NAME = "llm/local-rag-router"


def test_retrieval(client: Client, datasite: str, query: str):
    """
    Send a retrieval request to an LLM through the Syft LLM Router.

    Parameters
    ----------
    client : Client
        The Syft client instance.
    datasite : str
        The datasite to send the request to.
    query : str
        The query to send to the model.
    """

    # Create request
    request_data = {
        "query": query,
        # "options": {"limit": 3},
    }

    try:
        # Send the request
        future = rpc.send(
            client=client,
            url=rpc.make_url(datasite=datasite, app_name=APP_NAME, endpoint="retrieve"),
            body=request_data,
            expiry="5m",
            cache=True,
        )
        response = future.wait()
        response.raise_for_status()
        try:
            retrieval_response = response.model(RetrievalResponse)
            return retrieval_response.results
        except Exception:
            return response.model(RouterError)
    except Exception as e:
        logger.error(f"Error sending retrieval request: {e}")
        return None


# Example usage
if __name__ == "__main__":
    import time
    import sys

    if len(sys.argv) == 2:
        query = sys.argv[1]
        datasite = None  # Will use client.email
    elif len(sys.argv) == 3:
        query = sys.argv[1]
        datasite = sys.argv[2]
    else:
        print("Usage: python chat_test.py \"query\" [email]")
        print("  - query: The query to send to the model (required)")
        print("  - email: The datasite email (optional, defaults to your own datasite)")
        sys.exit(1)

    client = Client.load()
    # Use provided email as datasite, otherwise use client.email
    if datasite is None:
        datasite = client.email
    start = time.time()
    print("Test retrieval....")
    response = test_retrieval(
        client=client,
        datasite=datasite,
        query=query,
    )
    print(response)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
