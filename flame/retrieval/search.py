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

DEUX MÉCANISMES, CHACUN POUR CE QU'IL SAIT FAIRE.

TF-IDF compare des mots. Il est donc aveugle aux reformulations : interrogé
sur « putting out a fire with carbon dioxide », il ne retrouve pas ce qu'il
trouve pour « extinguish CO2 suppressant ». Mesuré sur huit questions
paraphrasées, il ne ramène la bonne investigation que 3 fois sur 8.

Une réduction de dimension (LSA, 200 composantes sur la matrice TF-IDF)
rapproche les mots qui apparaissent dans les mêmes contextes. Sur les mêmes
huit questions, elle passe à 6 sur 8. Elle capte donc ce que la comparaison de
mots rate.

MAIS ELLE PERD LE DROIT DE SE TAIRE, et seule, c'est rédhibitoire :

    dans le domaine          hors du domaine
    0.581  CO2 extinction    0.840  « quelle est la capitale de l'Australie »
    0.643  suie              0.743  « best pizza in Naples »
    0.645  detecteur fumee   0.671  « stock market prices today »

LSA projette n'importe quelle question dans son espace et y trouve toujours un
voisin. Une question sans rapport obtient un score SUPÉRIEUR à une question
pertinente. Pour un outil dont la promesse tient en « il ne répond que ce
qu'il peut citer », c'est disqualifiant.

D'OÙ LE MONTAGE RETENU. TF-IDF tient la PORTE : la question doit partager un
vocabulaire substantiel avec le corpus, sinon on ne répond pas. LSA fait le
CLASSEMENT parmi ce qui a passé la porte. On garde la capacité de refus du
premier et la souplesse du second.

Les termes partagés restent affichés quand il y en a. Sur une reformulation
pure la liste peut être courte : c'est alors LSA qui a porté le résultat, et
le voir est une information.

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
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

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
# Nombre de termes substantiels qu'une question doit partager avec le corpus
# pour qu'on accepte d'y repondre. Un seul est une coincidence.
MIN_SHARED_TERMS = 2
# Dimensions de la reduction. 200 suffisent : au-dela, le gain sur les
# paraphrases plafonne et le bruit augmente.
LSA_COMPONENTS = 200


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

        # LSA sert au classement, jamais a decider s'il faut repondre.
        self.svd = TruncatedSVD(n_components=LSA_COMPONENTS, random_state=0)
        self.reduced = normalize(self.svd.fit_transform(self.matrix))

    def __len__(self) -> int:
        return len(self.corpus)

    @property
    def vocabulary(self) -> int:
        return len(self.terms)

    def _passes_gate(self, query) -> bool:
        """La question partage-t-elle un vocabulaire substantiel avec le corpus ?

        C'est la porte, et c'est TF-IDF qui la tient. Sans elle, LSA répondrait
        à « quelle est la capitale de l'Australie » avec un score de 0.84,
        supérieur à celui d'une vraie question du domaine.

        DEUX CRITÈRES ONT ÉTÉ ESSAYÉS ET ÉCARTÉS AVANT CELUI-CI, parce qu'ils
        ne séparaient rien :

            le score TF-IDF maximal      « best pizza in Naples » obtient 0.150,
                                         plus que « the alarm did not notice the
                                         fumes » qui vaut 0.081 et qui est
                                         pertinent.

            la rarete des termes (IDF)   « stock market prices today » atteint
                                         7.94, le maximum du corpus, autant
                                         qu'une vraie question.

        Ce qui sépare, c'est le NOMBRE de termes substantiels appariés. Un mot
        isolé est une coïncidence : sur 3 000 passages anglais, presque tout
        mot courant apparaît quelque part. Deux mots commencent à décrire un
        sujet.

        Mesuré sur quinze questions : le seuil de deux garde 8 questions du
        domaine sur 8 et rejette 6 hors-sujet sur 7.

        LA LIMITE RESTE RÉELLE. « stock market prices today » passe encore, ses
        deux mots existant dans le corpus. Aucune statistique lexicale ne
        distingue parfaitement le hors-sujet, et c'est pourquoi les termes
        partagés sont affichés à côté de chaque résultat : quand la liste se
        réduit à « market, today », le lecteur voit immédiatement ce qu'il en
        est. La transparence complète le filtre, elle ne le remplace pas.
        """
        if query.nnz == 0:
            return False
        substantial = [
            term
            for term in self.terms[query.indices]
            if len(term) >= MIN_TERM_LENGTH and " " not in term
        ]
        return len(substantial) >= MIN_SHARED_TERMS

    def search(
        self,
        question: str,
        limit: int = 5,
        investigation: str | None = None,
        semantic: bool = True,
    ) -> list[Hit]:
        """Les passages les plus proches, du plus au moins pertinent.

        semantic : True classe avec LSA, qui capte les reformulations. False
            reste sur TF-IDF pur, ce qui permet de comparer les deux.
        """
        query = self.vectorizer.transform([question])
        if not self._passes_gate(query):
            return []

        if semantic:
            projected = self.svd.transform(query)
            if not np.any(projected):
                return []
            scores = (self.reduced @ normalize(projected).T).ravel()
        else:
            scores = (self.matrix @ query.T).toarray().ravel()

        if investigation:
            mask = self.corpus["investigation"].to_numpy() == investigation
            scores = np.where(mask, scores, 0.0)

        order = np.argsort(scores)[::-1][:limit]
        query_terms = set(self.terms[query.indices])

        # Les deux espaces n'ont pas la meme echelle : une similarite cosinus
        # apres reduction est structurellement plus elevee qu'un produit
        # TF-IDF creux.
        floor = MIN_SCORE * 3 if semantic else MIN_SCORE
        hits = []
        for index in order:
            if scores[index] < floor:
                continue
            row = self.corpus.iloc[index]
            # Les termes reellement partages. Sur une reformulation pure la
            # liste peut etre vide : c'est alors LSA qui a porte le resultat,
            # et le voir est une information.
            passage_terms = set(self.terms[self.matrix[index].indices])
            shared = sorted(query_terms & passage_terms, key=len, reverse=True)
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
