"""Wrapper léger autour d'un tokenizer Hugging Face de type BPE.

Par défaut on cible `camembert-base` (SentencePiece BPE entraîné sur du FR).
Le wrapper accepte n'importe quel `PreTrainedTokenizer` injecté, ce qui
permet de tester sans télécharger de modèle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

DEFAULT_TOKENIZER_NAME = "camembert-base"
DEFAULT_MAX_LENGTH = 256


class _HFTokenizerLike(Protocol):
    def __call__(
        self,
        text,
        padding,
        truncation,
        max_length,
        return_attention_mask: bool,
        return_tensors: Optional[str],
    ): ...

    def decode(self, ids, skip_special_tokens: bool) -> str: ...


@dataclass(frozen=True)
class EncodedBatch:
    input_ids: List[List[int]]
    attention_mask: List[List[int]]

    def __len__(self) -> int:
        return len(self.input_ids)


class BPETokenizer:
    """Façade minimaliste sur un tokenizer HF.

    Args:
        tokenizer: instance déjà chargée. Si None, on tente
                   `AutoTokenizer.from_pretrained(model_name)`.
        model_name: nom HF utilisé si `tokenizer` est None.
        max_length: longueur max appliquée au padding/troncation.
    """

    def __init__(
        self,
        tokenizer: Optional[_HFTokenizerLike] = None,
        model_name: str = DEFAULT_TOKENIZER_NAME,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        if tokenizer is None:
            from transformers import AutoTokenizer  # import paresseux

            tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tok = tokenizer
        self._max_length = max_length

    @property
    def max_length(self) -> int:
        return self._max_length

    def encode(self, texts: Sequence[str]) -> EncodedBatch:
        """Encode un batch de textes en IDs + attention masks."""
        if isinstance(texts, str):
            texts = [texts]
        out = self._tok(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_attention_mask=True,
            return_tensors=None,
        )
        return EncodedBatch(
            input_ids=list(out["input_ids"]),
            attention_mask=list(out["attention_mask"]),
        )

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)
