"""Extrait le texte des rapports NASA en passages citables.

    py scripts/extract_pdf_text.py

POURQUOI DES PASSAGES ET NON DES DOCUMENTS ENTIERS.

Une recherche qui renvoie « la réponse est dans ce rapport de 227 pages » ne
sert à rien. L'unité utile est le paragraphe : assez court pour être lu, assez
long pour porter un sens, et surtout rattachable à une page précise.

Chaque passage conserve donc son investigation, son document et sa page. C'est
ce qui permettra à la recherche de répondre AVEC SES SOURCES plutôt que de se
prononcer toute seule.

CE QUI EST LU, ET CE QUI NE L'EST PAS.

On ne lit que les dossiers `reports/`, jamais `raw/`. Les deux contiennent les
mêmes fichiers — `reports/` est la copie de travail dédoublonnée — et lire les
deux produirait chaque passage en double, ce qui fausserait toute recherche
par similarite.

LES DIAPOSITIVES SONT GARDÉES MALGRÉ LEUR FAIBLE DENSITÉ. Un rapport donne
environ 2 000 caractères par page, une présentation entre 400 et 600. Elles
restent utiles : une diapositive dit souvent en une phrase ce qu'un rapport
développe en trois pages. Leur densité est enregistrée pour qu'on sache d'où
vient un passage court.

LIMITES, À DIRE PLUTÔT QU'À DÉCOUVRIR. L'extraction d'un PDF perd la mise en
page : les tableaux deviennent des suites de nombres sans en-têtes, les
formules se disloquent, les colonnes se mélangent parfois. Ce corpus sert à
retrouver un PASSAGE À LIRE, pas à extraire des valeurs. Pour des chiffres,
les tables de `data/processed/` sont la source.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flame.common.paths import GROUND, MICROGRAVITY, PROCESSED  # noqa: E402

warnings.filterwarnings("ignore")

# Un passage plus court ne porte pas de sens ; plus long ne se lit plus d'un
# coup d'oeil et dilue la recherche.
MIN_CHARS = 180
MAX_CHARS = 1400

# Un PDF converti en texte produit aussi du remplissage : tables des matieres
# reduites a des points de suite, colonnes de mesures detachees de leurs
# en-tetes. Ces passages ne se lisent pas et polluent la recherche — un
# sommaire contient tous les mots-cles du document sans rien en dire.
MAX_DIGIT_SHARE = 0.35


def _clean(text: str) -> str:
    """Répare ce que l'extraction d'un PDF casse systématiquement."""
    # Cesures de fin de ligne : « combus-\ntion » redevient « combustion ».
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # Retours a la ligne internes a un paragraphe.
    text = re.sub(r"(?<![.!?:;])\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _is_filler(block: str) -> bool:
    """Un sommaire ou une colonne de chiffres n'est pas un passage a lire."""
    if len(re.findall(r"\.\s*\.\s*\.", block)) > 3:
        return True
    digits = sum(character.isdigit() for character in block)
    return digits / max(len(block), 1) > MAX_DIGIT_SHARE


def _passages(text: str) -> list[str]:
    """Découpe en paragraphes, en recoupant ceux qui sont trop longs."""
    chunks = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if len(block) < MIN_CHARS or _is_filler(block):
            continue
        while len(block) > MAX_CHARS:
            cut = block.rfind(". ", 0, MAX_CHARS)
            if cut < MIN_CHARS:
                cut = MAX_CHARS
            chunks.append(block[: cut + 1].strip())
            block = block[cut + 1 :].strip()
        if len(block) >= MIN_CHARS:
            chunks.append(block)
    return [c for c in chunks if not _is_filler(c)]


def extract() -> pd.DataFrame:
    import pypdf

    rows = []
    reports = sorted(
        list(MICROGRAVITY.glob("*/reports/*.pdf")) + list(GROUND.glob("*/reports/*.pdf"))
    )
    print(f"{len(reports)} rapports a lire\n")

    for path in reports:
        investigation = path.parent.parent.name
        try:
            reader = pypdf.PdfReader(str(path))
        except Exception as error:
            print(f"  ECHEC  {path.name[:48]} : {str(error)[:50]}")
            continue

        kept = 0
        for number, page in enumerate(reader.pages, start=1):
            try:
                raw = page.extract_text() or ""
            except Exception:
                continue
            for passage in _passages(_clean(raw)):
                rows.append(
                    {
                        "investigation": investigation,
                        "document": path.name,
                        "page": number,
                        "text": passage,
                    }
                )
                kept += 1
        print(f"  {investigation:9} {len(reader.pages):4d} p  {kept:4d} passages  {path.name[:46]}")

    return pd.DataFrame(rows)


def main() -> None:
    corpus = extract()
    destination = PROCESSED / "reports_corpus.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(destination, index=False, encoding="utf-8")

    size_mb = destination.stat().st_size / 1_048_576
    print(f"\n{'=' * 70}")
    print(f"  {len(corpus):,} passages  ·  {corpus['document'].nunique()} documents  "
          f"·  {corpus['investigation'].nunique()} investigations")
    print(f"  ecrit : {destination}  ({size_mb:.1f} Mo)")
    print(f"  longueur mediane d'un passage : {corpus['text'].str.len().median():.0f} caracteres")
    print("\n  passages par investigation :")
    counts = corpus["investigation"].value_counts()
    for name, count in counts.items():
        print(f"    {name:10} {count:5d}")


if __name__ == "__main__":
    main()
