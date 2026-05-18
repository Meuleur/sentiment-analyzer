"""Preprocessing texte pour le français.

Pipeline minimaliste, conçu pour la baseline TF-IDF et utilisable
en amont d'un tokenizer transformer.

Étapes :
  - mise en minuscules
  - normalisation Unicode NFKC
  - suppression des URL, mentions, hashtags (rarement utiles pour le sentiment)
  - dé-doublement des espaces et caractères répétés (>2 fois)
  - normalisation des accents *optionnelle* (utile pour la baseline TF-IDF,
    à éviter pour un transformer français qui exploite les accents).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")
_WHITESPACE_RE = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    """Retire les diacritiques (é → e, ç → c, ñ → n)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_text(text: str, *, remove_accents: bool = False) -> str:
    """Nettoie un texte français.

    Args:
        text: chaîne brute.
        remove_accents: si True, applique `strip_accents` à la fin.

    Returns:
        Texte normalisé (minuscules, sans URL/mention/hashtag, espaces compactés).
    """
    if not isinstance(text, str):
        raise TypeError(f"clean_text attend une str, reçu {type(text).__name__}")

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(" ", text)
    text = _REPEAT_CHAR_RE.sub(r"\1\1", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if remove_accents:
        text = strip_accents(text)
    return text


def clean_corpus(texts: Iterable[str], *, remove_accents: bool = False) -> List[str]:
    """Applique `clean_text` à un corpus."""
    return [clean_text(t, remove_accents=remove_accents) for t in texts]
