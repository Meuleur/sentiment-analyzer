"""Tests CLI + persistance — pipeline réel, entrée mockée."""

import io
import json
from pathlib import Path

import pytest

from src.baseline import train_baseline
from src.cli import predict, run
from src.persistence import load_model, save_model


POS = [
    "Film magnifique, j'ai adoré.",
    "Une pépite, à voir absolument.",
    "Sublime et émouvant.",
    "Très belle réalisation.",
]
NEG = [
    "Film ennuyeux, je me suis endormi.",
    "Scénario incompréhensible.",
    "Vraiment décevant, à éviter.",
    "Affreux, mal joué.",
]


@pytest.fixture
def trained_model_path(tmp_path: Path) -> Path:
    pipe = train_baseline(POS + NEG, [1] * len(POS) + [0] * len(NEG))
    path = tmp_path / "baseline.pkl"
    save_model(pipe, path)
    return path


def test_save_and_load_roundtrip(trained_model_path: Path):
    model = load_model(trained_model_path)
    preds = model.predict(["film magnifique", "vraiment mauvais"])
    assert len(preds) == 2


def test_load_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.pkl")


def test_predict_returns_named_labels(trained_model_path: Path):
    model = load_model(trained_model_path)
    preds = predict(model, ["magnifique film", "horrible film"])
    assert len(preds) == 2
    assert all(p.label_name in {"positive", "negative"} for p in preds)


def test_cli_text_format_with_args(trained_model_path: Path):
    stdout = io.StringIO()
    code = run(
        ["--model", str(trained_model_path), "Film magnifique, j'adore"],
        stdin=io.StringIO(""),
        stdout=stdout,
    )
    assert code == 0
    output = stdout.getvalue().strip()
    assert output.split("\t")[0] in {"positive", "negative"}


def test_cli_json_format(trained_model_path: Path):
    stdout = io.StringIO()
    code = run(
        ["--model", str(trained_model_path), "--format", "json", "Film magnifique"],
        stdin=io.StringIO(""),
        stdout=stdout,
    )
    assert code == 0
    parsed = json.loads(stdout.getvalue().strip())
    assert parsed["text"] == "Film magnifique"
    assert "label_name" in parsed


def test_cli_reads_stdin_when_no_args(trained_model_path: Path):
    stdout = io.StringIO()
    stdin = io.StringIO("Film magnifique\nFilm horrible\n")
    code = run(["--model", str(trained_model_path)], stdin=stdin, stdout=stdout)
    assert code == 0
    lines = stdout.getvalue().strip().splitlines()
    assert len(lines) == 2


def test_cli_empty_input_returns_error(trained_model_path: Path, capsys):
    code = run(
        ["--model", str(trained_model_path)],
        stdin=io.StringIO(""),
        stdout=io.StringIO(),
    )
    assert code == 2
