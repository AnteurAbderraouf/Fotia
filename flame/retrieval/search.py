"""Recherche dans les rapports NASA — répondre avec ses sources, ou se taire.

CE QUE CETTE COUCHE EST, ET CE QU'ELLE N'EST PAS.

Le §9 du handoff appelle ça la couche de « grounding » : ce qui transforme une
probabilité nue en quelque chose qu'un ingénieur accepterait de croire. Le
mécanisme est simple et il doit le rester.

    ce que c'est     une recherche par similarite qui renvoie les PASSAGES
                     les plus proches d'une question, chacun avec son
                     document et sa page

    ce que ce n'est  un modele qui repond de lui-meme. Un modele de langage
    pas              interroge sur la combustion en microgravite produira des
                     chiffres plausibles et invérifiables, impossibles a
                     distinguer des vrais. Dans un outil de securite
                     incendie, c'est disqualifiant.

La règle tient en une phrase : **il ne répond que ce qu'il peut citer.** S'il
ne trouve rien, il le dit.

POURQUOI TF-IDF PLUTÔT QUE DES EMBEDDINGS, POUR COMMENCER.

Un modèle d'embeddings capterait mieux les reformulations — « éteindre » et
« extinction » seraient rapprochés sans partager un mot. C'est un vrai
avantage, et c'est la suite logique.

Mais TF-IDF a une propriété que les embeddings n'ont pas : **on peut voir
pourquoi un passage a été retenu.** Chaque résultat affiche les termes qui ont
porté la correspondance. Sur un corpus technique où le vocabulaire est stable
— « extinction », « flame », « microgravity » reviennent tels quels — cette
lisibilité vaut plus que la souplesse, et elle permet de juger la recherche
au lieu de lui faire confiance.

Les bigrammes sont inclus, faute de quoi « cool flame » se réduirait à deux
mots indépendants dont le second est partout.

LIMITE DU CORPUS, À CONNAÎTRE AVANT DE CHERCHER. Onze investigations sur
vingt-quatre ont des rapports ; les autres n'en ont pas, et aucune recherche
ne fera apparaître ce qui n'a pas été publié. L'extraction perd par ailleurs
la mise en page : pour un chiffre, les tables de `data/processed/` sont la
source, pas ce corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from flame.common.paths import PROCESSED

CORPUS = PROCESSED / "reports_corpus.csv"

# En dessous, la correspondance ne repose que sur des mots trop communs pour
# vouloir dire quelque chose. Mieux vaut ne rien renvoyer.
MIN_SCORE = 0.08
# Un terme de trois lettres ou moins ne distingue rien. Une question posee en
# francais a ainsi trouve des passages anglais par le seul mot « la », que le
# filtre de mots vides anglais ne retire pas : le score etait honorable, la
# correspondance nulle. On exige donc au moins un terme substantiel partage.
MIN_TERM_LENGTH = 4


@dataclass
class Hit:
    """Un passage trouvé, avec de quoi le retrouver dans le PDF."""

    investigation: str
    document: str
    page: int
    text: str
    score: float
    terms: list[str]

    @property
    def citation(self) -> str:
        return f"{self.investigation} · {self.document}, page {self.page}"


class ReportSearch:
    """Recherche par similarité sur les passages extraits des rapports."""

    def __init__(self, corpus: pd.DataFrame | None = None):
        if corpus is None:
            if not CORPUS.exists():
                raise FileNotFoundError(
                    f"{CORPUS.name} absent. Le produire avec "
                    "`py scripts/extract_pdf_text.py`."
                )
            corpus = pd.read_csv(CORPUS)
        self.corpus = corpus.reset_index(drop=True)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            # Les bigrammes sont indispensables : « cool flame » n'est pas
            # « cool » plus « flame », et « flame » seul est partout.
            ngram_range=(1, 2),
            # Un terme present dans un seul passage ne generalise pas ; un
            # terme present partout ne distingue rien.
            min_df=2,
            max_df=0.55,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.corpus["text"])
        self.terms = np.array(self.vectorizer.get_feature_names_out())

    def __len__(self) -> int:
        return len(self.corpus)

    @property
    def vocabulary(self) -> int:
        return len(self.terms)

    def search(
        self,
        question: str,
        limit: int = 5,
        investigation: str | None = None,
    ) -> list[Hit]:
        """Les passages les plus proches, du plus au moins pertinent."""
        query = self.vectorizer.transform([question])
        if query.nnz == 0:
            return []

        scores = (self.matrix @ query.T).toarray().ravel()

        if investigation:
            mask = self.corpus["investigation"].to_numpy() == investigation
            scores = np.where(mask, scores, 0.0)

        order = np.argsort(scores)[::-1][:limit]
        query_terms = set(self.terms[query.indices])

        hits = []
        for index in order:
            if scores[index] < MIN_SCORE:
                continue
            row = self.corpus.iloc[index]
            # Les termes reellement partages : c'est ce qui rend la
            # correspondance verifiable plutot que magique.
            passage_terms = set(self.terms[self.matrix[index].indices])
            shared = sorted(query_terms & passage_terms, key=len, reverse=True)
            if not any(len(term) >= MIN_TERM_LENGTH for term in shared):
                continue
            hits.append(
                Hit(
                    investigation=row["investigation"],
                    document=row["document"],
                    page=int(row["page"]),
                    text=row["text"],
                    score=float(scores[index]),
                    terms=shared[:6],
                )
            )
        return hits

    def coverage(self) -> pd.DataFrame:
        """Ce que le corpus contient, pour savoir ce qu'il ne contient pas."""
        grouped = (
            self.corpus.groupby("investigation")
            .agg(passages=("text", "size"), documents=("document", "nunique"))
            .sort_values("passages", ascending=False)
            .reset_index()
        )
        return grouped


def main() -> None:
    engine = ReportSearch()
    print("=" * 74)
    print("RECHERCHE DANS LES RAPPORTS NASA")
    print("=" * 74)
    print(
        f"\n{len(engine):,} passages · {engine.corpus['document'].nunique()} documents "
        f"· {engine.corpus['investigation'].nunique()} investigations"
        f"\nvocabulaire retenu : {engine.vocabulary:,} termes et bigrammes\n"
    )

    questions = [
        "why does carbon dioxide extinguish a droplet flame",
        "what is a cool flame and when does it appear",
        "smoke detector response to different materials",
        "how does gravity change the shape of a flame",
        "recette de tarte aux pommes",
    ]
    for question in questions:
        print("-" * 74)
        print(f"Q : {question}")
        hits = engine.search(question, limit=2)
        if not hits:
            print("    aucun passage assez proche — la question sort du corpus.")
            continue
        for hit in hits:
            print(f"\n  [{hit.score:.3f}] {hit.citation}")
            print(f"  termes partages : {', '.join(hit.terms)}")
            body = hit.text[:280].replace("\n", " ")
            print(f"  « {body}... »")
        print()


if __name__ == "__main__":
    main()
