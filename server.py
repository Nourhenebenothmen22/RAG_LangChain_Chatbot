"""Server entrypoint for Kortex AI Hybrid RAG FastAPI application."""

import argparse
import uvicorn


def run_server() -> None:
    parser = argparse.ArgumentParser(description="Start Hybrid RAG API server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"Starting Kortex AI Hybrid RAG API server on http://{args.host}:{args.port}")
    uvicorn.run("rag_langchain_chatbot.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    run_server()
