"""NER module.

Identifies entity mentions with a spaCy model trained on OntoNotes and keeps
only linkable entity types. For each mention it returns the surface form,
character offsets, entity type and the sentence it occurs in (the sentence is
later used as context by the retriever). A leading "the" is removed from the
surface form and the offsets are recomputed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import spacy

# OntoNotes types that denote values rather than real-world or abstract entities.
NON_LINKABLE_TYPES = frozenset(
    {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}
)


@dataclass
class Mention:
    text: str            # span.text (surface form)
    start_offset: int    # character offset in the input text
    end_offset: int      # character offset in the input text
    type: str            # OntoNotes label
    sentence: str        # sentence containing the mention
    sentence_id: int     # index of the sentence in the input text

    def to_dict(self) -> dict:
        return asdict(self)


class NER:
    def __init__(
        self,
        model_name: str = "en_core_web_lg",
        excluded_types: frozenset[str] | set[str] = NON_LINKABLE_TYPES,
    ):
        self.model_name = model_name
        self.nlp = spacy.load(model_name)
        self.excluded_types = set(excluded_types)

    def extract(self, text: str) -> list[Mention]:
        doc = self.nlp(text)
        sentence_ids = {sent.start: i for i, sent in enumerate(doc.sents)}

        mentions = []
        for ent in doc.ents:
            if ent.label_ in self.excluded_types:
                continue
            sent = ent.sent
            surface_form, start = ent.text, ent.start_char
            if surface_form.lower().startswith("the "):
                surface_form, start = surface_form[4:], start + 4
            mentions.append(
                Mention(
                    text=surface_form,
                    start_offset=start,
                    end_offset=ent.end_char,
                    type=ent.label_,
                    sentence=sent.text.strip(),
                    sentence_id=sentence_ids[sent.start],
                )
            )
        return mentions
