"""Graph re-ranker module.

Builds an undirected, unweighted graph whose nodes are the candidates of a
group of mentions (e.g. one sentence) and whose edges link two candidates when
one points to the other. Two strategies:

- "wikidata": an edge A-B exists if B appears as a value in A's claims (or the
  reverse). Since the claims of every candidate are fetched, outgoing claims
  are enough to find every Wikidata link between two candidates.
- "wikidata+wikipedia": the Wikidata edges fused with the links found in each
  candidate's Wikipedia page (MediaWiki API, links mapped to QIDs).

Only edges between two candidates are kept, self-loops are removed and edge
types are ignored. Candidates are then re-ranked with
final_score = cosine_score + centrality_score.
"""
from __future__ import annotations

import networkx as nx
import requests

from src.ner import Mention
from src.retriever import Candidate, get_json, make_session

STRATEGIES = ("wikidata", "wikidata+wikipedia")
CENTRALITIES = ("degree", "betweenness", "eigenvector")


class GraphReranker:
    def __init__(
        self,
        strategy: str = "wikidata",
        centrality: str = "degree",
        language: str = "en",
        session: requests.Session | None = None,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if centrality not in CENTRALITIES:
            raise ValueError(f"centrality must be one of {CENTRALITIES}")
        self.strategy = strategy
        self.centrality = centrality
        self.wikipedia_api = f"https://{language}.wikipedia.org/w/api.php"
        self.session = session or make_session()
        self._links_cache: dict[str, set[str]] = {}

    # ---------- Wikipedia links ----------

    def wikipedia_links(self, title: str) -> set[str]:
        """QIDs of the pages linked by a Wikipedia page (main namespace only)."""
        if title in self._links_cache:
            return self._links_cache[title]
        params = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "titles": title,
            "generator": "links",
            "gplnamespace": 0,
            "gpllimit": "max",
            "prop": "pageprops",
            "ppprop": "wikibase_item",
            "redirects": 1,
        }
        qids = set()
        while True:
            data = get_json(self.session, self.wikipedia_api, params)
            for page in data.get("query", {}).get("pages", []):
                qid = page.get("pageprops", {}).get("wikibase_item")
                if qid:
                    qids.add(qid)
            if "continue" not in data:
                break
            params.update(data["continue"])
        self._links_cache[title] = qids
        return qids

    # ---------- Graph ----------

    def build_graph(self, candidate_lists: list[list[Candidate]]) -> nx.Graph:
        unique = {c.qid: c for candidates in candidate_lists for c in candidates}
        nodes = set(unique)

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for qid, candidate in unique.items():
            neighbors = set(candidate.wikidata_neighbors)
            if self.strategy == "wikidata+wikipedia" and candidate.wikipedia_title:
                neighbors |= self.wikipedia_links(candidate.wikipedia_title)
            for other in neighbors & nodes:
                if other != qid:
                    graph.add_edge(qid, other)
        return graph

    def compute_centrality(self, graph: nx.Graph) -> dict[str, float]:
        if graph.number_of_edges() == 0:
            return {n: 0.0 for n in graph}
        if self.centrality == "degree":
            scores = nx.degree_centrality(graph)
        elif self.centrality == "betweenness":
            scores = nx.betweenness_centrality(graph)
        else:
            try:
                scores = nx.eigenvector_centrality(graph, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                return {n: 0.0 for n in graph}
 
        # Divide by the maximum value in the sentence graph: the most central
        # candidate gets 1 whatever the measure, isolated candidates get 0.
        max_score = max(scores.values())
        if max_score == 0:
            return {n: 0.0 for n in graph}
        return {n: scores[n] / max_score if graph.degree(n) > 0 else 0.0 for n in graph}

    
    # ---------- Re-ranking ----------

    def rerank(
        self, mentions: list[Mention], candidate_lists: list[list[Candidate]]
    ) -> tuple[list[list[Candidate]], nx.Graph]:
        """Re-rank the candidates of a group of mentions; returns the lists and the graph."""
        graph = self.build_graph(candidate_lists)
        centrality = self.compute_centrality(graph)

        labels, mentions_of = {}, {}
        for mention, candidates in zip(mentions, candidate_lists):
            for c in candidates:
                labels[c.qid] = c.label
                mentions_of.setdefault(c.qid, set()).add(mention.text)

        reranked = []
        for candidates in candidate_lists:
            for c in candidates:
                c.centrality_score = float(centrality.get(c.qid, 0.0))
                c.final_score = c.cosine_score + c.centrality_score
                c.evidence = [
                    f"linked to {labels[n]} ({n}), candidate for "
                    + ", ".join(f"'{m}'" for m in sorted(mentions_of[n]))
                    for n in sorted(graph.neighbors(c.qid))
                ]
            reranked.append(sorted(candidates, key=lambda c: c.final_score, reverse=True))
        return reranked, graph
