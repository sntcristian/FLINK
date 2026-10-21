"""Pipeline module.

Runs the sub-modules on an input sentence:
NER -> candidate retrieval -> (optional) sentence-transformer re-ranking
-> (optional) graph re-ranking.

The configuration (trf_rerank, trf_model, graph_rerank, strategy, centrality)
is given when the pipeline is created, so that the spaCy and transformer
models are loaded once. The sentence and its id are given when the pipeline
is called: pipeline(sentence, sentence_id).

Retrieved candidates can be stored in a JSON file (candidates_path), keyed by
surface form. When the file exists, it is loaded at start-up and the Wikidata
API is only called for surface forms that are not in it. Call
save_candidates() at the end of a run to write the file. The file assumes
that all runs use the same language and limit.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from src.graph_reranker import GraphReranker
from src.ner import NER, Mention
from src.retriever import Candidate, Retriever, make_session

# Candidate fields produced by retrieval (scores and evidence are recomputed at every run)
RETRIEVAL_FIELDS = ("qid", "label", "description", "aliases", "wikipedia_title", "search_rank")


def candidate_to_json(candidate: Candidate) -> dict:
    data = {field: getattr(candidate, field) for field in RETRIEVAL_FIELDS}
    data["wikidata_neighbors"] = sorted(candidate.wikidata_neighbors)
    return data


def candidate_from_json(data: dict) -> Candidate:
    return Candidate(**{**data, "wikidata_neighbors": set(data["wikidata_neighbors"])})


class Pipeline:
    def __init__(
        self,
        trf_rerank: bool = True,
        trf_model: str = "dmis-lab/UltraBERT-Embed-150M",
        graph_rerank: bool = True,
        strategy: str = "wikidata",
        centrality: str = "degree",
        ner_model: str = "en_core_web_lg",
        language: str = "en",
        limit: int = 20,
        candidates_path: str | Path | None = None,
    ):
        """
        trf_rerank:      re-rank candidates with a sentence-transformer
        trf_model:       Hugging Face id of the sentence-transformer re-ranker
        graph_rerank:    re-rank candidates with graph context (skipped when fewer
                         than two mentions of the sentence have candidates)
        strategy:        "wikidata" or "wikidata+wikipedia"
        centrality:      "degree", "betweenness" or "eigenvector"
        ner_model:       spaCy model trained on OntoNotes
        language:        language of the input text and of Wikidata labels
        limit:           number of candidates retrieved per mention
        candidates_path: JSON file of retrieved candidates (loaded if it exists)
        """
        self.trf_rerank = trf_rerank
        self.trf_model = trf_model if trf_rerank else None
        self.graph_rerank = graph_rerank
        self.strategy = strategy if graph_rerank else None
        self.centrality = centrality if graph_rerank else None
        self.ner_model = ner_model

        session = make_session()
        self.ner = NER(ner_model)
        self.retriever = Retriever(
            model_name=self.trf_model, language=language, limit=limit, session=session
        )
        self.graph_reranker = (
            GraphReranker(strategy=strategy, centrality=centrality, language=language, session=session)
            if graph_rerank
            else None
        )

        # Retrieved candidates by surface form, as JSON-serialisable dicts
        self.candidates_path = Path(candidates_path) if candidates_path else None
        self.candidates_cache: dict[str, list[dict]] = {}
        if self.candidates_path and self.candidates_path.exists():
            self.candidates_cache = json.loads(self.candidates_path.read_text(encoding="utf-8"))

    @property
    def config(self) -> dict:
        return {
            "ner_model": self.ner_model,
            "trf_rerank": self.trf_rerank,
            "trf_model": self.trf_model,
            "graph_rerank": self.graph_rerank,
            "strategy": self.strategy,
            "centrality": self.centrality,
        }

    def retrieve(self, mention: Mention) -> list[Candidate]:
        """Candidates of a mention, from the cache when its surface form was already retrieved."""
        if mention.text not in self.candidates_cache:
            self.candidates_cache[mention.text] = [
                candidate_to_json(c) for c in self.retriever.retrieve(mention)
            ]
        # New Candidate objects at every call: re-ranking writes scores on them.
        return [candidate_from_json(data) for data in self.candidates_cache[mention.text]]

    def save_candidates(self, path: str | Path | None = None) -> None:
        """Writes the retrieved candidates to path (default: candidates_path)."""
        path = Path(path) if path else self.candidates_path
        if path is None:
            raise ValueError("No path given and no candidates_path set")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.candidates_cache, ensure_ascii=False, indent=1), encoding="utf-8")

    def __call__(self, sentence: str, sentence_id: int | str) -> dict:
        # 1. Mentions. The whole input sentence is the context of each mention.
        mentions = [
            replace(m, sentence=sentence, sentence_id=sentence_id)
            for m in self.ner.extract(sentence)
        ]

        # 2. Candidates, ordered by the API or by cosine similarity.
        candidate_lists = []
        for mention in mentions:
            candidates = self.retrieve(mention)
            if self.trf_rerank:
                candidates = self.retriever.rerank(mention, candidates)
            candidate_lists.append(candidates)

        # 3. Graph re-ranking, only when at least two mentions have candidates.
        graph_applied = (
            self.graph_reranker is not None and sum(1 for c in candidate_lists if c) > 1
        )
        edges = []
        if graph_applied:
            candidate_lists, graph = self.graph_reranker.rerank(mentions, candidate_lists)
            edges = sorted(sorted(edge) for edge in graph.edges())

        return {
            "sentence_id": sentence_id,
            "sentence": sentence,
            "config": self.config,
            "graph_reranked": graph_applied,
            "graph_edges": edges,
            "mentions": [
                {
                    "text": m.text,
                    "start_offset": m.start_offset,
                    "end_offset": m.end_offset,
                    "type": m.type,
                    "prediction": candidates[0].qid if candidates else None,
                    "candidates": [c.to_dict() for c in candidates],
                }
                for m, candidates in zip(mentions, candidate_lists)
            ],
        }