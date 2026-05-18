"""Tests du pipeline de preprocessing FR."""

import pytest

from src.preprocessing import clean_corpus, clean_text, strip_accents


def test_lowercase_and_strip():
    assert clean_text("   Bonjour LE Monde  ") == "bonjour le monde"


def test_removes_url():
    out = clean_text("Voir https://example.com/article super cool")
    assert "http" not in out
    assert "super cool" in out


def test_removes_mentions_and_hashtags():
    out = clean_text("Salut @jean, regarde #cinema c'est top")
    assert "@jean" not in out
    assert "#cinema" not in out
    assert "c'est top" in out


def test_collapses_repeated_chars():
    assert clean_text("trooop biiiien!!!!") == "troop biien!!"


def test_keeps_accents_by_default():
    out = clean_text("Élève à l'école")
    assert "é" in out
    assert "à" in out


def test_strip_accents_helper():
    assert strip_accents("Élève à l'école") == "Eleve a l'ecole"
    assert strip_accents("garçon naïf") == "garcon naif"


def test_clean_text_with_accent_removal():
    out = clean_text("Élève à l'école", remove_accents=True)
    assert "é" not in out
    assert "à" not in out
    assert out == "eleve a l'ecole"


def test_normalizes_unicode_compat_forms():
    # "ﬁ" (U+FB01) → "fi" via NFKC
    assert clean_text("eﬃcace") == "efficace"


def test_clean_corpus_keeps_order():
    raw = ["Premier!", "DEUXIÈME texte", "  troisième  "]
    out = clean_corpus(raw)
    assert out == ["premier!", "deuxième texte", "troisième"]


def test_clean_text_rejects_non_string():
    with pytest.raises(TypeError):
        clean_text(123)  # type: ignore[arg-type]
