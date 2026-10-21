"""Retriever module.

1. Retrieves candidates for a surface form with wbsearchentities.
2. Fetches labels, descriptions, aliases, the Wikipedia sitelink and the claims
   of each candidate with wbgetentities (batches of 50 ids). The claims are
   reduced to the set of items the candidate points to; the graph_reranker
   uses them to build the Wikidata subgraph.
3. Re-ranks the candidates with a sentence-transformer (cosine similarity
   between the query and "{label}: {description}"). The "query" prompt is
   applied to the query only, never to the candidate documents.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from src.ner import Mention

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Wikimedia APIs reject generic User-Agents. Put a real contact address here:
# https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy
USER_AGENT = "FLINK/0.1 (contact: c.santini12@unimc.it)"

QUERY_TEMPLATE = (
    "Query: {surface_form}. Context: {sentence}."
)

@dataclass
class Candidate:
    qid: str
    label: str
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    wikipedia_title: str | None = None
    wikidata_neighbors: set[str] = field(default_factory=set)  # items in the candidate's claims
    search_rank: int = 0           # position returned by wbsearchentities
    cosine_score: float = 0.0
    centrality_score: float = 0.0
    final_score: float = 0.0
    evidence: list[str] = field(default_factory=list)

    @property
    def uri(self) -> str:
        return f"http://www.wikidata.org/entity/{self.qid}"

    @property
    def document(self) -> str:
        return f"{self.label}: {self.description}" if self.description else self.label

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("wikidata_neighbors")  # large, not useful in the output
        d["uri"] = self.uri
        return d


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def get_json(session: requests.Session, url: str, params: dict, retries: int = 3, timeout: int = 30) -> dict:
    """GET with a simple exponential backoff on rate limiting and server errors."""
    for attempt in range(retries):
        response = session.get(url, params=params, timeout=timeout)
        if response.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        return response.json()
    return {}


def claim_item_values(claims: dict) -> set[str]:
    """QIDs appearing as values of the candidate's statements (edge types are ignored)."""
    values = set()
    for statements in claims.values():
        for statement in statements:
            snak = statement.get("mainsnak", {})
            if snak.get("snaktype") == "value" and snak.get("datatype") == "wikibase-item":
                values.add(snak["datavalue"]["value"]["id"])
    return values


class Retriever:
    def __init__(
        self,
        model_name: str = "Snowflake/snowflake-arctic-embed-m-v2.0",
        language: str = "en",
        limit: int = 10,
        session: requests.Session | None = None,
    ):
        self.model_name = model_name
        self.language = language
        self.limit = limit
        self.session = session or make_session()
        # model_name=None: retrieval only, no sentence-transformer is loaded.
        self.model = SentenceTransformer(model_name) if model_name else None
        # Use the model's "query" prompt when it defines one (Arctic does);
        # other models in the ablation are used without a prompt.
        self.query_prompt = (
            "query" if self.model is not None and "query" in (self.model.prompts or {}) else None
        )
        self._search_cache: dict[str, list[dict]] = {}
        self._entity_cache: dict[str, dict] = {}

    # ---------- Wikidata API ----------

    def search(self, label: str) -> list[dict]:
        if label not in self._search_cache:
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": label,
                "srnamespace": 0,        # main namespace = items
                "srlimit": self.limit,
                "srprop": "",            # only the titles (QIDs) are needed
                "uselang": self.language,
            }
            results = get_json(self.session, WIKIDATA_API, params).get("query", {}).get("search", [])
            # On Wikidata the title of an item page is its QID
            self._search_cache[label] = [{"id": r["title"]} for r in results]
        return self._search_cache[label]


    def get_entities(self, qids: list[str]) -> dict[str, dict]:
        missing = [q for q in dict.fromkeys(qids) if q not in self._entity_cache]
        for i in range(0, len(missing), 50):
            params = {
                "action": "wbgetentities",
                "format": "json",
                "ids": "|".join(missing[i:i + 50]),
                "props": "labels|descriptions|aliases|sitelinks|claims",
                "languages": self.language,
                "sitefilter": f"{self.language}wiki",
            }
            data = get_json(self.session, WIKIDATA_API, params)
            self._entity_cache.update(data.get("entities", {}))
        return {q: self._entity_cache[q] for q in qids if q in self._entity_cache}

    def _to_candidate(self, result: dict, entity: dict, rank: int) -> Candidate:
        lang = self.language
        return Candidate(
            qid=result["id"],
            label=entity.get("labels", {}).get(lang, {}).get("value") or result.get("label", result["id"]),
            description=entity.get("descriptions", {}).get(lang, {}).get("value") or result.get("description", ""),
            aliases=[a["value"] for a in entity.get("aliases", {}).get(lang, [])],
            wikipedia_title=entity.get("sitelinks", {}).get(f"{lang}wiki", {}).get("title"),
            wikidata_neighbors=claim_item_values(entity.get("claims", {})),
            search_rank=rank,
        )

    def retrieve(self, mention: Mention) -> list[Candidate]:
        results = self.search(mention.text)
        entities = self.get_entities([r["id"] for r in results])
        return [
            self._to_candidate(r, entities.get(r["id"], {}), rank)
            for rank, r in enumerate(results)
            if "missing" not in entities.get(r["id"], {})
        ]

    # ---------- Sentence-transformer re-ranking ----------

    def build_query(self, mention: Mention) -> str:
        return QUERY_TEMPLATE.format(sentence=mention.sentence, surface_form=mention.text)

    def rerank(self, mention: Mention, candidates: list[Candidate]) -> list[Candidate]:
        if self.model is None:
            raise ValueError("Retriever was created without a sentence-transformer model")
        if not candidates:
            return candidates
        query_emb = self.model.encode(
            [self.build_query(mention)], prompt_name=self.query_prompt, normalize_embeddings=True
        )
        doc_embs = self.model.encode([c.document for c in candidates], normalize_embeddings=True)
        scores = doc_embs @ query_emb[0]
        for candidate, score in zip(candidates, np.atleast_1d(scores)):
            candidate.cosine_score = float(score)
            candidate.final_score = candidate.cosine_score
        return sorted(candidates, key=lambda c: c.cosine_score, reverse=True)

    def retrieve_and_rerank(self, mention: Mention) -> list[Candidate]:
        return self.rerank(mention, self.retrieve(mention))
