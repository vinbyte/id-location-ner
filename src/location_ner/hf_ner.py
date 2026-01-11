"""Optional HuggingFace NER wrapper.

We keep this separate from the core extractor so that transformers/torch remain
optional dependencies.

This module extracts entities from text using a pretrained HuggingFace model,
then returns only location-like entities.

Why chunking is required
------------------------
Most BERT-style NER models have a maximum sequence length of 512 tokens.
If we send very long text directly into the pipeline, the model can crash with
errors like:

    RuntimeError: The size of tensor a (...) must match ... (512)

To make this robust for long, messy inputs (scrapes, multi-paragraph text), we
run the model on overlapping *character chunks* and then merge/deduplicate the
results.

Design goals:
- Never crash on long text
- Preserve correct start/end character offsets w.r.t. the original text
- Keep the implementation readable and testable

NOTE:
- HuggingFace label sets differ across models.
- The example model `cahya/bert-base-indonesian-NER` commonly uses labels like:
  - LOC for location
  - PER for person
  - ORG for organization
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HfEntity:
    """A single entity prediction from a HuggingFace NER pipeline."""

    label: str
    text: str
    start_char: int
    end_char: int
    score: float


class HuggingFaceNer:
    """Runs a HuggingFace NER model (pipeline) and extracts location entities."""

    def __init__(
        self,
        model_name: str,
        *,
        max_chunk_chars: int = 1800,
        chunk_overlap_chars: int = 250,
    ):
        """Create the HuggingFace NER wrapper.

        Args:
            model_name: Model name (HuggingFace hub id or local path).
            max_chunk_chars: Max chunk size in characters.
                This is a practical proxy to stay well below the model's 512 token limit.
            chunk_overlap_chars: Overlap between consecutive chunks in characters.
                Overlap reduces the chance of splitting an entity across chunk boundaries.
        """
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be > 0")
        if chunk_overlap_chars < 0:
            raise ValueError("chunk_overlap_chars must be >= 0")
        if chunk_overlap_chars >= max_chunk_chars:
            raise ValueError("chunk_overlap_chars must be < max_chunk_chars")

        self.model_name = model_name
        self.max_chunk_chars = int(max_chunk_chars)
        self.chunk_overlap_chars = int(chunk_overlap_chars)

        # Lazy import keeps transformers/torch optional.
        from transformers import pipeline  # type: ignore

        # aggregation_strategy="simple" merges wordpiece tokens into a span.
        self._pipe = pipeline(
            task="ner",
            model=model_name,
            aggregation_strategy="simple",
        )

    def extract_locations(self, text: str) -> list[HfEntity]:
        """Return location entities predicted by the model.

        This method is safe for very long texts.

        Args:
            text: Input text.

        Returns:
            A list of HfEntity objects (location-only), with offsets relative to
            the original `text`.
        """
        entities: list[HfEntity] = []

        for chunk_text, chunk_start in _iter_text_chunks(
            text,
            max_chunk_chars=self.max_chunk_chars,
            overlap_chars=self.chunk_overlap_chars,
        ):
            # NOTE: we intentionally do NOT enable tokenizer truncation here.
            # Chunking keeps the text safely under the model limit.
            results = self._pipe(chunk_text)

            for r in results:
                label = (r.get("entity_group") or r.get("entity") or "").upper()

                # Heuristic: accept common location labels.
                if label not in {"LOC", "LOCATION", "GPE"}:
                    continue

                start = int(r["start"]) + chunk_start
                end = int(r["end"]) + chunk_start

                entities.append(
                    HfEntity(
                        label=label,
                        text=r["word"],
                        start_char=start,
                        end_char=end,
                        score=float(r["score"]),
                    )
                )

        return _dedupe_entities(entities)


def _iter_text_chunks(
    text: str,
    *,
    max_chunk_chars: int,
    overlap_chars: int,
):
    """Yield (chunk_text, chunk_start_char) pairs.

    Implementation is character-based for simplicity and stable offset mapping.

    We attempt to end chunks at a reasonable boundary (newline/space/punctuation)
    near the max length.
    """
    n = len(text)
    if n == 0:
        return

    start = 0
    while start < n:
        target_end = min(n, start + max_chunk_chars)
        end = _find_chunk_end(text, start, target_end)
        if end <= start:
            end = target_end

        yield text[start:end], start

        if end >= n:
            break

        # Step forward with overlap.
        start = max(0, end - overlap_chars)


def _find_chunk_end(text: str, start: int, target_end: int) -> int:
    """Find a good chunk boundary <= target_end.

    We search backward in a small window for a separator to reduce splitting.
    """
    if target_end <= start:
        return start

    # Search back within this many characters.
    window = 250
    search_start = max(start, target_end - window)

    # Prefer sentence-ish boundaries first, then whitespace.
    preferred = {"\n", ".", "!", "?", ";", ","}

    for i in range(target_end - 1, search_start - 1, -1):
        if text[i] in preferred:
            return i + 1

    for i in range(target_end - 1, search_start - 1, -1):
        if text[i].isspace():
            return i + 1

    return target_end


def _dedupe_entities(entities: list[HfEntity]) -> list[HfEntity]:
    """Deduplicate entities from overlapping chunks.

    Strategy:
    - Sort by (start, -length) to make overlap checks stable.
    - Consider two entities duplicates if:
      - same label
      - overlapping character spans
      - normalized text matches (case-insensitive)
    - Keep the one with higher score, with a tie-break preferring longer spans.
    """
    if not entities:
        return []

    def norm(s: str) -> str:
        return " ".join(s.lower().split())

    entities_sorted = sorted(
        entities, key=lambda e: (e.start_char, -(e.end_char - e.start_char))
    )

    kept: list[HfEntity] = []
    for ent in entities_sorted:
        if not kept:
            kept.append(ent)
            continue

        last = kept[-1]
        if (
            ent.label == last.label
            and _spans_overlap(
                (ent.start_char, ent.end_char), (last.start_char, last.end_char)
            )
            and norm(ent.text) == norm(last.text)
        ):
            if (ent.score > last.score) or (
                ent.score == last.score
                and (ent.end_char - ent.start_char) > (last.end_char - last.start_char)
            ):
                kept[-1] = ent
            continue

        kept.append(ent)

    return kept


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]
