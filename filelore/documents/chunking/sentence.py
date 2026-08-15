"""Lightweight Unicode-aware sentence boundary detection."""

from __future__ import annotations


class UnicodeSentenceSplitter:
    """Split common Latin, CJK, and Arabic sentence punctuation."""

    _STRONG_TERMINATORS = frozenset("!?！？؟。…")
    _CLOSERS = frozenset("\"'”’»)]}")
    _ABBREVIATIONS = frozenset(
        {
            "dr",
            "e.g",
            "etc",
            "fig",
            "i.e",
            "mr",
            "mrs",
            "ms",
            "no",
            "prof",
            "sn",
            "st",
            "vb",
            "vs",
            "örn",
        }
    )

    def split(self, text: str) -> tuple[str, ...]:
        """Return non-empty sentences while retaining terminal punctuation."""

        normalized = " ".join(text.replace("\x00", "").split())
        if not normalized:
            return ()

        sentences: list[str] = []
        sentence_start = 0
        index = 0
        while index < len(normalized):
            character = normalized[index]
            boundary_end: int | None = None

            if character in self._STRONG_TERMINATORS:
                boundary_end = self._terminator_end(normalized, index)
            elif character == "." and self._period_ends_sentence(
                normalized,
                index,
            ):
                boundary_end = self._terminator_end(normalized, index)

            if boundary_end is None:
                index += 1
                continue

            sentence = normalized[sentence_start:boundary_end].strip()
            if sentence:
                sentences.append(sentence)
            sentence_start = boundary_end
            while (
                sentence_start < len(normalized)
                and normalized[sentence_start].isspace()
            ):
                sentence_start += 1
            index = sentence_start

        remainder = normalized[sentence_start:].strip()
        if remainder:
            sentences.append(remainder)
        return tuple(sentences)

    def _terminator_end(self, text: str, index: int) -> int:
        end = index + 1
        terminators = self._STRONG_TERMINATORS | {"."}
        while end < len(text) and text[end] in terminators:
            end += 1
        while end < len(text) and text[end] in self._CLOSERS:
            end += 1
        return end

    def _period_ends_sentence(self, text: str, index: int) -> bool:
        previous = text[index - 1] if index else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        if previous.isdigit() and following.isdigit():
            return False
        if following and not following.isspace() and following not in self._CLOSERS:
            return False
        if self._preceding_token(text, index) in self._ABBREVIATIONS:
            return False
        return True

    @staticmethod
    def _preceding_token(text: str, index: int) -> str:
        start = index - 1
        while start >= 0 and (text[start].isalpha() or text[start] == "."):
            start -= 1
        return text[start + 1 : index].casefold()
