# sentiment-analyzer

Classifier des phrases françaises en **positif / négatif / neutre**.

Le projet part d'une baseline classique (TF-IDF + régression logistique) et progresse
jusqu'à un transformer fine-tuné (CamemBERT), exposé derrière une API FastAPI.

## Objectifs

- Comparer plusieurs approches sur le même dataset (baseline → embeddings → transformer)
- Mesurer honnêtement (accuracy, F1 macro) — pas de cherry-picking
- Fournir une CLI et une API minimales pour tester un modèle entraîné

## Structure

```
sentiment-analyzer/
├── README.md
├── requirements.txt
├── src/                # code source (load_data, preprocessing, baseline, ...)
├── tests/              # tests pytest
└── data/               # données (raw/processed, ignorées par git)
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
pytest -q
```

## Statut

En cours. Roadmap globale : [Meuleur/AIproject](https://github.com/Meuleur/AIproject).

Part of [Meuleur/AIproject](https://github.com/Meuleur/AIproject).
