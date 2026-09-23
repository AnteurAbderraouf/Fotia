"""Compare les deux mécanismes de recherche sur des questions paraphrasées.

    py scripts/benchmark_search.py

POURQUOI CE BANC D'ESSAI EXISTE.

Comparer deux recherches « à l'œil » ne prouve rien : on retient les cas qui
confirment ce qu'on espérait. Ce script pose donc huit questions dont on
connaît l'investigation attendue, et compte.

Les questions sont délibérément des PARAPHRASES. Elles ne contiennent pas le
vocabulaire technique des rapports : « the alarm did not notice the fumes »
plutôt que « smoke detector response ». C'est exactement là que la comparaison
de mots doit échouer et que la réduction de dimension doit aider.

L'investigation attendue est déduite du SUJET, pas des mots. Le test reste
donc valable pour les deux méthodes.

CE QUE CE BANC NE MESURE PAS. Il dit si la bonne investigation apparaît, pas
si le passage renvoyé est le meilleur de cette investigation. C'est une mesure
grossière, et elle suffit pour trancher entre deux mécanismes ; elle ne
suffirait pas pour régler finement l'un d'eux.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flame.retrieval.search import ReportSearch  # noqa: E402

# (question paraphrasee, investigations acceptables)
CASES = [
    ("putting out a fire with carbon dioxide", {"PSI-69", "PSI-39", "PSI-117"}),
    ("how quickly the fuel bead shrinks", {"PSI-39", "PSI-117"}),
    ("invisible low temperature burning", {"PSI-39", "PSI-117", "PSI-159"}),
    ("the alarm did not notice the fumes", {"PSI-101", "PSI-102"}),
    ("weightlessness changes the fire shape", {"PSI-115", "PSI-10", "PSI-159"}),
    ("how much soot does this fuel make", {"PSI-107"}),
    ("burning a plastic sheet aboard the station", {"PSI-25", "PSI-98", "PSI-100"}),
    ("oxygen level below which nothing burns", {"PSI-69", "PSI-39", "PSI-10"}),
]

OUT_OF_CORPUS = [
    "recette de tarte aux pommes",
    "quelle est la capitale de l'Australie",
    "best pizza in Naples",
    "stock market prices today",
    "how to train a dog",
]


def main() -> None:
    engine = ReportSearch()
    print("=" * 76)
    print("BANC D'ESSAI DE LA RECHERCHE")
    print("=" * 76)
    print(f"\n{len(engine):,} passages, {engine.vocabulary:,} termes\n")

    print("1. QUESTIONS PARAPHRASEES — la bonne investigation est-elle dans le top 3 ?\n")
    print(f"   {'question':46}{'TF-IDF':>10}{'hybride':>10}")
    print("   " + "-" * 66)
    literal = semantic = 0
    for question, expected in CASES:
        found_literal = any(
            hit.investigation in expected
            for hit in engine.search(question, limit=3, semantic=False)
        )
        found_semantic = any(
            hit.investigation in expected
            for hit in engine.search(question, limit=3, semantic=True)
        )
        literal += found_literal
        semantic += found_semantic
        print(
            f"   {question[:44]:46}{'oui' if found_literal else 'non':>10}"
            f"{'oui' if found_semantic else 'non':>10}"
        )
    print("   " + "-" * 66)
    print(f"   {'total':46}{f'{literal}/{len(CASES)}':>10}{f'{semantic}/{len(CASES)}':>10}")

    print("\n2. QUESTIONS HORS CORPUS — la recherche sait-elle se taire ?\n")
    refused = 0
    for question in OUT_OF_CORPUS:
        hits = engine.search(question)
        refused += not hits
        verdict = "se tait" if not hits else f"repond ({len(hits)})"
        print(f"   {question[:44]:46}{verdict:>20}")
    print(f"\n   {refused}/{len(OUT_OF_CORPUS)} refus corrects")

    print("\n" + "=" * 76)
    print(
        "LECTURE\n"
        f"\n  Le classement par reduction de dimension gagne "
        f"{semantic - literal} question(s) sur les paraphrases."
        "\n  La porte tenue par TF-IDF conserve le refus : LSA seule donnait"
        "\n  0.84 a « quelle est la capitale de l'Australie », soit plus qu'a"
        "\n  la plupart des vraies questions du domaine."
        "\n\n  C'est pour cela que les deux sont montees en serie plutot qu'en"
        "\n  concurrence : l'une decide s'il faut repondre, l'autre dans quel"
        "\n  ordre."
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
