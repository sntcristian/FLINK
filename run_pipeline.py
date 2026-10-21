import argparse
import sys

from src.pipeline import Pipeline


def print_result(result: dict, top_k: int = 3) -> None:
    print(f"\n[{result['sentence_id']}] {result['sentence']}")
    print(f"    graph re-ranking applied: {result['graph_reranked']}, edges: {result['graph_edges']}")
    for m in result["mentions"]:
        print(f"  '{m['text']}' ({m['type']}, {m['start_offset']}-{m['end_offset']}) -> {m['prediction'] or 'NIL'}")
        for c in m["candidates"][:top_k]:
            print(
                f"      {c['qid']:<11} {c['label'][:35]:<35} "
                f"cos={c['cosine_score']:.3f} cent={c['centrality_score']:.3f} "
                f"final={c['final_score']:.3f}  {c['description'][:60]}"
            )
            for evidence in c["evidence"]:
                print(f"          evidence: {evidence}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Text-to-Wikidata entity linking.")
    p.add_argument("text", help="input text")
    p.add_argument("--trf-rerank", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--trf-model", default="dmis-lab/UltraBERT-Embed-150M")
    p.add_argument("--graph-rerank", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strategy", default="wikidata", choices=["wikidata", "wikidata+wikipedia"])
    p.add_argument("--centrality", default="degree", choices=["degree", "betweenness", "eigenvector"])
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    pipeline = Pipeline(
        trf_rerank=args.trf_rerank,
        trf_model=args.trf_model,
        graph_rerank=args.graph_rerank,
        strategy=args.strategy,
        centrality=args.centrality,
        limit=args.limit,
    )
    print("config:", pipeline.config)
    print_result(pipeline(args.text, 0), top_k=args.top_k)