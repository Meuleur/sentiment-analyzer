"""Tests du wrapper BPETokenizer — backend HF mocké pour éviter les downloads."""

from src.tokenization import BPETokenizer, EncodedBatch


class _FakeHFTokenizer:
    """Mime juste assez l'API HF : pad un batch à la max length du batch."""

    SPECIAL_BOS = 0
    SPECIAL_EOS = 2
    PAD = 1

    def __call__(self, texts, padding, truncation, max_length, return_attention_mask, return_tensors):
        # Tokenisation jouet : un id par caractère ASCII, capé à max_length-2 pour BOS/EOS.
        seqs = []
        for t in texts:
            ids = [self.SPECIAL_BOS] + [ord(c) for c in t][: max_length - 2] + [self.SPECIAL_EOS]
            seqs.append(ids)

        target = max(len(s) for s in seqs)
        input_ids = []
        attention_mask = []
        for s in seqs:
            pad_n = target - len(s)
            input_ids.append(s + [self.PAD] * pad_n)
            attention_mask.append([1] * len(s) + [0] * pad_n)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens):
        chars = []
        for i in ids:
            if skip_special_tokens and i in (self.SPECIAL_BOS, self.SPECIAL_EOS, self.PAD):
                continue
            chars.append(chr(i))
        return "".join(chars)


def _make_tok(max_length: int = 32) -> BPETokenizer:
    return BPETokenizer(tokenizer=_FakeHFTokenizer(), max_length=max_length)


def test_encode_returns_encoded_batch():
    tok = _make_tok()
    out = tok.encode(["bonjour", "salut"])
    assert isinstance(out, EncodedBatch)
    assert len(out) == 2
    assert len(out.input_ids) == 2
    assert len(out.attention_mask) == 2


def test_encode_pads_to_longest_in_batch():
    tok = _make_tok()
    out = tok.encode(["a", "abcdef"])
    # toutes les séquences doivent avoir la même longueur
    lengths = {len(seq) for seq in out.input_ids}
    assert len(lengths) == 1
    # le mask du plus court doit contenir au moins un 0 (padding)
    assert 0 in out.attention_mask[0]
    # le plus long n'est pas paddé
    assert 0 not in out.attention_mask[1]


def test_encode_respects_max_length():
    tok = _make_tok(max_length=6)
    out = tok.encode(["a" * 100])
    assert len(out.input_ids[0]) == 6


def test_encode_accepts_single_string():
    tok = _make_tok()
    out = tok.encode("hello")
    assert len(out) == 1


def test_decode_roundtrip_strips_specials():
    tok = _make_tok()
    out = tok.encode(["abc"])
    text = tok.decode(out.input_ids[0])
    assert text == "abc"


def test_max_length_property():
    tok = _make_tok(max_length=42)
    assert tok.max_length == 42
