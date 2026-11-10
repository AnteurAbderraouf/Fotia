"""Inventaire vérifiable de l'archive NASA, celle que git ne transporte pas.

LE PROBLEME.

Le dépôt pèse 12 Mo et passe par GitHub. L'archive NASA pèse 4,9 Go et reste
sur le disque : 25 zips d'origine, l'arbre déplié `raw/`, les rapports PDF,
les grilles CFD. Quand on change de machine, cette archive voyage par clé USB,
disque externe ou réseau, et une copie de cette taille peut perdre un fichier
ou en tronquer un sans rien dire. Un zip tronqué ne se voit qu'au moment où on
essaie de l'ouvrir, c'est-à-dire des semaines plus tard.

LA SOLUTION, ET POURQUOI ELLE TIENT.

Ce script écrit `archive_manifest.csv` : un chemin, une taille et une somme
SHA-256 par fichier. Ce manifeste est VERSIONNE, lui. Il voyage donc par git,
pendant que les données voyagent par le disque, et les deux se rejoignent sur
la nouvelle machine :

    ancien PC     py scripts/archive_manifest.py --write
                  puis copier combustion_science/ sur le disque externe

    nouveau PC    git clone ...
                  recopier combustion_science/ depuis le disque
                  py scripts/archive_manifest.py --verify

La vérification nomme ce qui manque, ce qui est de la mauvaise taille et ce
qui est corrompu. Elle ne dit pas « ça a l'air bon ».

CE QUI EST INVENTORIE : ce que git ignore, d'après git lui-même, et non
d'après une liste recopiée ici qui aurait divergé au premier changement de
`.gitignore`. Sont écartés les caches Python et `data/figures/`, qui se
régénèrent et dont le contenu change à chaque exécution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "archive_manifest.csv"

# Regenerables ou sans interet : les inventorier ferait du bruit a chaque
# execution sans rien proteger.
SKIP = ("__pycache__/", "data/figures/", ".pyc", ".tmp", ".bak")

CHUNK = 1 << 20  # 1 Mio


class Progress:
    """Avancement lisible dans un terminal comme dans un journal.

    Le retour chariot n'efface la ligne que sur un vrai terminal. Rediriger la
    sortie vers un fichier produisait sinon une ligne par fichier, soit 511
    lignes de bruit. Hors terminal, on n'ecrit donc qu'a chaque dixieme.
    """

    def __init__(self, total: int):
        self.total = max(total, 1)
        self.done = 0
        self.live = sys.stdout.isatty()
        self.step = 0

    def advance(self, size: int, label: str) -> None:
        self.done += size
        share = self.done / self.total
        if self.live:
            print(f"\r  {share:5.1%}  {label[:58]:58s}", end="", flush=True)
            return
        if share >= (self.step + 1) / 10:
            self.step = int(share * 10)
            print(f"  {share:5.0%}")

    def close(self) -> None:
        print("\n" if self.live else "")


def archive_files(required: bool = True) -> list[Path]:
    """Les fichiers que git ignore, d'après git.

    `required=False` renvoie une liste vide au lieu d'echouer quand git est
    absent. La verification s'en sert : sur une machine neuve, git n'est pas
    forcement installe au moment ou l'on controle la copie, et ce serait
    absurde de perdre le controle des 502 fichiers pour un listage annexe.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--others", "--ignored", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, FileNotFoundError):
        if required:
            raise SystemExit(
                "git est introuvable. L'inventaire s'appuie sur "
                "`git ls-files`, donc l'ecriture du manifeste demande git."
            )
        return []
    if result.returncode != 0:
        if required:
            raise SystemExit(f"git a echoue : {result.stderr.strip()}")
        return []

    found = []
    for line in result.stdout.splitlines():
        relative = line.strip()
        if not relative or any(part in relative for part in SKIP):
            continue
        path = ROOT / relative
        if path.is_file():
            found.append(path)
    return sorted(found)


def digest(path: Path) -> str:
    """SHA-256 en flux, pour ne pas charger un zip de 300 Mo en memoire."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            sha.update(block)
    return sha.hexdigest()


def human(size: int) -> str:
    return f"{size / 1048576:.0f} Mo" if size >= 1048576 else f"{size / 1024:.0f} Ko"


def write(quick: bool) -> int:
    files = archive_files()
    total = sum(p.stat().st_size for p in files)
    print(f"{len(files)} fichiers, {human(total)} a inventorier")
    if quick:
        print("mode rapide : tailles seules, aucune somme de controle\n")
    else:
        print("lecture complete pour les sommes SHA-256, patience\n")

    progress = Progress(total)
    rows = []
    for path in files:
        size = path.stat().st_size
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": size,
                "sha256": "" if quick else digest(path),
            }
        )
        progress.advance(size, path.name)
    progress.close()

    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{MANIFEST.name} ecrit : {len(rows)} fichiers, {human(total)}")
    print("Il est versionne : `git add archive_manifest.csv` puis commit.")
    return 0


def verify(quick: bool) -> int:
    if not MANIFEST.exists():
        raise SystemExit(
            f"{MANIFEST.name} absent. Le produire sur la machine d'origine "
            "avec `py scripts/archive_manifest.py --write`."
        )

    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        expected = {row["path"]: row for row in csv.DictReader(handle)}

    total = sum(int(row["bytes"]) for row in expected.values())
    hashed = any(row["sha256"] for row in expected.values())
    print(f"{len(expected)} fichiers attendus, {human(total)}")
    if quick or not hashed:
        print("tailles seules" if quick else "manifeste sans sommes, tailles seules")
    print()

    missing: list[str] = []
    wrong_size: list[str] = []
    corrupt: list[str] = []
    progress = Progress(total)

    for relative, row in expected.items():
        path = ROOT / relative
        size = int(row["bytes"])
        progress.advance(size, Path(relative).name)

        if not path.is_file():
            missing.append(relative)
            continue
        actual = path.stat().st_size
        if actual != size:
            wrong_size.append(f"{relative} : {human(actual)} au lieu de {human(size)}")
            continue
        if not quick and row["sha256"] and digest(path) != row["sha256"]:
            corrupt.append(relative)

    progress.close()
    listed = archive_files(required=False)
    extra = sorted(
        {p.relative_to(ROOT).as_posix() for p in listed} - set(expected)
    ) if listed else []

    for title, items in [
        ("MANQUANTS", missing),
        ("MAUVAISE TAILLE", wrong_size),
        ("CORROMPUS, la somme ne correspond pas", corrupt),
    ]:
        if items:
            print(f"{title} ({len(items)}) :")
            for item in items[:25]:
                print(f"  {item}")
            if len(items) > 25:
                print(f"  ... et {len(items) - 25} autres")
            print()

    if extra:
        # Pas une erreur : un fichier en plus peut etre un ajout legitime fait
        # apres l'ecriture du manifeste.
        print(f"EN PLUS, absents du manifeste ({len(extra)}) :")
        for item in extra[:10]:
            print(f"  {item}")
        if len(extra) > 10:
            print(f"  ... et {len(extra) - 10} autres")
        print()

    broken = len(missing) + len(wrong_size) + len(corrupt)
    if broken:
        print(f"{broken} fichier(s) en defaut. La copie est incomplete.")
        return 1
    print(f"Archive complete et intacte : {len(expected)} fichiers, {human(total)}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true",
                       help="ecrire le manifeste depuis le disque courant")
    group.add_argument("--verify", action="store_true",
                       help="comparer le disque courant au manifeste")
    parser.add_argument("--quick", action="store_true",
                        help="tailles seules, sans somme de controle")
    args = parser.parse_args()
    return write(args.quick) if args.write else verify(args.quick)


if __name__ == "__main__":
    sys.exit(main())
