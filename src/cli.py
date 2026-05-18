"""CLI d'inférence pour la baseline sentiment-analyzer.

Usage :
    # prédire sur des arguments
    python -m src.cli --model runs/baseline.pkl "Film magnifique" "Ennuyeux"

    # prédire sur stdin (une phrase par ligne)
    cat reviews.txt | python -m src.cli --model runs/baseline.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, TextIO

from src.persistence import load_model
from src.preprocessing import clean_text

LABEL_NAMES = {0: "negative", 1: "positive"}


@dataclass(frozen=True)
class Prediction:
    text: str
    label: int
    label_name: str

    def to_json(self) -> str:
        return json.dumps(
            {"text": self.text, "label": self.label, "label_name": self.label_name},
            ensure_ascii=False,
        )


def predict(model, texts: Sequence[str]) -> List[Prediction]:
    """Renvoie une `Prediction` par texte d'entrée."""
    cleaned = [clean_text(t, remove_accents=True) for t in texts]
    raw_preds = model.predict(cleaned)
    return [
        Prediction(text=t, label=int(p), label_name=LABEL_NAMES.get(int(p), str(p)))
        for t, p in zip(texts, raw_preds)
    ]


def _read_stdin(stream: TextIO) -> List[str]:
    return [line.rstrip("\n") for line in stream if line.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentiment-cli",
        description="Predict sentiment with a saved baseline model.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the pickled sklearn pipeline.",
    )
    parser.add_argument(
        "texts",
        nargs="*",
        help="Texts to classify. If omitted, reads one text per line on stdin.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    return parser


def _collect_texts(args: argparse.Namespace, stdin: TextIO) -> List[str]:
    if args.texts:
        return list(args.texts)
    return _read_stdin(stdin)


def _format(preds: Iterable[Prediction], fmt: str) -> str:
    if fmt == "json":
        return "\n".join(p.to_json() for p in preds)
    return "\n".join(f"{p.label_name}\t{p.text}" for p in preds)


def run(argv: Sequence[str], *, stdin: TextIO, stdout: TextIO) -> int:
    args = _build_parser().parse_args(argv)
    texts = _collect_texts(args, stdin)
    if not texts:
        print("error: no input texts provided", file=sys.stderr)
        return 2

    model = load_model(args.model)
    preds = predict(model, texts)
    stdout.write(_format(preds, args.format) + "\n")
    return 0


def main() -> None:  # pragma: no cover
    sys.exit(run(sys.argv[1:], stdin=sys.stdin, stdout=sys.stdout))


if __name__ == "__main__":  # pragma: no cover
    main()
