"""Main CLI entrypoint for the Hybrid RAG pipeline."""

import argparse
from typing import Optional

from rag_langchain_chatbot.config import get_settings
from rag_langchain_chatbot.pipeline import HybridRAGPipeline


def display_response(response, max_content_len: int = 400) -> None:
    """Pretty-print a HybridRAGResponse to the terminal."""
    print("\n" + "=" * 70)
    print(f"QUERY: {response.query}")
    print(f"STATUS: Retrieved {len(response.chunks)} chunks in {response.latency_ms:.2f} ms")
    print(f"RERANKED: {'Yes (Cross-Encoder)' if response.reranked else 'No (RRF only)'}")
    print("=" * 70)

    for chunk in response.chunks:
        score_type = "Cross-Encoder Score" if response.reranked else "RRF Score"
        print(f"\n[RANK {chunk.rank}] {score_type}: {chunk.score:.4f} (Dense: #{chunk.dense_rank or 'N/A'}, BM25: #{chunk.sparse_rank or 'N/A'}, RRF: {chunk.rrf_score or 0.0:.4f})")
        print(f"  Source:   {chunk.source}")
        loc = f"Page: {chunk.page + 1}" if chunk.page is not None else (f"Record: {chunk.record_id}" if chunk.record_id else "N/A")
        print(f"  Location: {loc}")
        print(f"  Chunk ID: {chunk.chunk_id[:16]}...")
        print("  Excerpt:")
        cleaned = " ".join(chunk.content.split())
        if len(cleaned) > max_content_len:
            cleaned = cleaned[:max_content_len] + "..."
        print(f"    \"{cleaned}\"")

    if getattr(response, "answer", None):
        print("\n" + "=" * 70)
        print("GENERATED ANSWER:")
        print("=" * 70)
        print(response.answer)
        print(f"\n[Model: {response.model_name} | Tokens: {response.total_tokens} (Prompt: {response.prompt_tokens}, Completion: {response.completion_tokens}) | Est. Cost: ${response.estimated_cost_usd:.6f}]")

    print("\n" + "-" * 70)
    print("FORMATTED CONTEXT PREVIEW:")
    print("-" * 70)
    preview = response.format_context()
    lines = preview.split("\n")
    print("\n".join(lines[:12]))
    if len(lines) > 12:
        print("    [... remaining context omitted ...]")


def interactive_mode(pipeline: HybridRAGPipeline, top_k: Optional[int], rerank: bool) -> None:
    """Start an interactive query session."""
    print("\nStarting Interactive Hybrid RAG session.")
    print("Type your query and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("Ask a question > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting session.")
                break

            response = pipeline.query(query, top_k=top_k, rerank=rerank)
            display_response(response)
            print()

        except (KeyboardInterrupt, EOFError):
            print("\nSession terminated.")
            break


def cli() -> None:
    """Parse CLI arguments and route to appropriate pipeline actions."""
    parser = argparse.ArgumentParser(
        description="High-Performance Hybrid RAG Pipeline with LangChain & Cross-Encoder Reranking."
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Run document ingestion and build persistent vector and BM25 indices."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="When indexing, force clear existing vector store and indices."
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Search query to run against the hybrid index."
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=None,
        help="Number of final results to retrieve (overrides config)."
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable Cross-Encoder reranking and return raw RRF results."
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive query session in the terminal."
    )

    args = parser.parse_args()
    settings = get_settings()
    pipeline = HybridRAGPipeline(settings=settings)

    if args.index:
        print(f"Starting indexing on {settings.data_dir} (force={args.force})...")
        stats = pipeline.index(force_reindex=args.force)
        print("Indexing completed successfully!")
        print(stats)
        return

    if args.interactive:
        interactive_mode(pipeline, top_k=args.top_k, rerank=not args.no_rerank)
        return

    # If specific query provided
    if args.query:
        response = pipeline.query(args.query, top_k=args.top_k, rerank=not args.no_rerank)
        display_response(response)
        return

    # Default demonstration
    demo_query = "What happens when the fraud score exceeds 0.6?"
    print("No arguments provided. Running demonstration query:")
    response = pipeline.query(demo_query, top_k=args.top_k, rerank=not args.no_rerank)
    display_response(response)


if __name__ == "__main__":
    cli()
