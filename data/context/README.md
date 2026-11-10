# data/context : donnees de contexte collectees a la main

Contrairement a `data/processed/` (jetable, regenere par `scripts/build_all.py`),
ces fichiers ne se regenerent PAS : ils ont ete releves dans des sources et
doivent etre versionnes.

| Fichier | Contenu |
|---|---|
| `contexte_incidents.csv` | feux de vaisseaux (Apollo 1, Mir, Bondarenko, Saffire) |
| `contexte_atmospheres.csv` | pression et % d'O2 par vaisseau |
| `contexte_gravite.csv` | regimes de gravite (Terre, Lune, Mars, ISS) |
| `literature_crosscheck.csv` | 14 resultats publies (conditions -> observation) pour valider les modeles |
| `msl1_dce_heptane.csv` | 38 gouttes d'heptane brulees sur Spacelab MSL-1 (1997), Table 1 de NASA/TM-2003-212553 ; 32 utilisables (colonne `usable`) nourrissent la constante K |
| `proprietes_combustibles.csv` | proprietes des carburants (NIST WebBook, PubChem/HSDB) pour la partie physique du moteur |

## Statut de verification

La colonne `verifie` des fichiers `contexte_*` vaut « non » : les citations
ont ete extraites par un outil de lecture et pas recoupees sur le texte brut.
Avant de s'en servir dans l'application, ouvrir l'URL de chaque ligne et
confirmer la citation, puis remplacer « non » par « oui, <nom>, <date> ».

`literature_crosscheck.csv` a ete verifie par script contre le texte des PDF
(chaque citation a ete retrouvee a la page indiquee). Deux cellules restent
incertaines, signalees dans la colonne `remarque` (lignes 4 et 6).

## Copie dans `nasa/Gravity context/`

Les `contexte_*.csv` existent aussi dans le dossier du designer. Le code lit
CE dossier-ci. Si une ligne est corrigee, la corriger ici (ou recopier).

## MSL-1

`msl1_dce_heptane.csv` a ete recopie a la main depuis le PDF
(https://ntrs.nasa.gov/citations/20040000941, p. 17-18). Avant de le citer comme verifie,
relire chaque ligne contre le tableau imprime.
