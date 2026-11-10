"""Mise en route sur une machine neuve, en une commande.

    py scripts/setup.py

CE QU'ELLE FAIT, ET CE QU'ELLE NE PEUT PAS FAIRE.

Elle installe les dépendances, vérifie que les tables se reconstruisent, que
les gabarits 3D sont valides, que le tableau de bord se rend dans tous ses
états, et elle contrôle l'archive NASA si elle est là.

Elle ne peut PAS faire apparaître l'archive. Les 5 Go de zips, de grilles CFD
et de rapports ne sont pas dans le dépôt et ne sont sur aucun serveur à nous :
ils se copient par disque externe, ou se re-téléchargent depuis psi.nasa.gov.
Ce script dit clairement s'ils sont là, et ce qui change quand ils n'y sont
pas, plutôt que de laisser la question ouverte.

L'ABSENCE DE L'ARCHIVE N'EST PAS UNE ERREUR. Le tableau de bord tourne
entièrement sans elle : les tables réduites sont versionnées. Ce qu'on perd,
c'est la possibilité de les RECALCULER depuis les champs bruts, donc de
modifier la façon dont on les résume.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# La version de developpement. En dessous de 3.11 la syntaxe employee dans le
# paquet ne passe pas, `X | Y` dans les annotations evaluees notamment.
MINIMUM = (3, 11)
DEVELOPED_ON = (3, 14)

BAR = "=" * 70


def run(label: str, command: list[str], optional: bool = False) -> bool:
    """Lance une etape et rapporte son issue d'une ligne."""
    print(f"\n{BAR}\n  {label}\n{BAR}")
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode == 0:
        print(f"\n  -> ok")
        return True
    verdict = "ignore" if optional else "ECHEC"
    print(f"\n  -> {verdict} (code {result.returncode})")
    return optional


def python_version() -> bool:
    current = sys.version_info[:3]
    print(f"{BAR}\n  Python\n{BAR}")
    print(f"  interpreteur : {sys.executable}")
    print(f"  version      : {'.'.join(map(str, current))}")
    if current[:2] < MINIMUM:
        print(
            f"\n  -> ECHEC : il faut au moins "
            f"{'.'.join(map(str, MINIMUM))}. Installer une version recente "
            "depuis python.org, puis relancer."
        )
        return False
    if current[:2] != DEVELOPED_ON:
        print(
            f"\n  -> ok, mais le projet a ete developpe sur "
            f"{'.'.join(map(str, DEVELOPED_ON))}. Un ecart de version peut "
            "faire bouger un affichage, pas un resultat."
        )
    else:
        print("\n  -> ok")
    return True


def archive_state() -> str:
    """Presente, partielle, ou absente."""
    manifest = ROOT / "archive_manifest.csv"
    if not manifest.exists():
        return "sans manifeste"
    fields = list(ROOT.glob("combustion_science/*/PSI-115/fields/*.csv"))
    zips = list(ROOT.glob("combustion_science/*/*/*.zip"))
    if fields and zips:
        return "presente"
    if fields or zips:
        return "partielle"
    return "absente"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--no-install", action="store_true",
        help="sauter pip, pour verifier une installation deja faite",
    )
    parser.add_argument(
        "--quick-archive", action="store_true",
        help="controler l'archive sur les tailles seules, sans somme SHA-256",
    )
    args = parser.parse_args()

    failures: list[str] = []

    if not python_version():
        return 1

    if not args.no_install:
        if not run(
            "Dependances",
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        ):
            failures.append("dependances")
            # Sans les paquets, la suite ne veut plus rien dire.
            print("\nInstallation echouee, arret ici.")
            return 1

    state = archive_state()
    if state == "presente":
        command = [sys.executable, "scripts/archive_manifest.py", "--verify"]
        if args.quick_archive:
            command.append("--quick")
        if not run("Archive NASA, 502 fichiers", command):
            failures.append("archive incomplete")
    else:
        print(f"\n{BAR}\n  Archive NASA\n{BAR}")
        print(f"  etat : {state}")
        print(
            "\n  -> absente, et ce n'est PAS une erreur. Le tableau de bord\n"
            "     tourne sans elle, les tables reduites etant versionnees.\n"
            "     Ce qui manque, c'est de quoi les RECALCULER : les 25 zips\n"
            "     NASA, l'arbre raw/, les grilles CFD et les rapports PDF.\n"
            "     Ils se copient par disque externe depuis l'ancienne\n"
            "     machine, ou se retelechargent depuis psi.nasa.gov."
        )

    for label, command in [
        ("Reconstruction des 10 tables", ["scripts/build_all.py"]),
        ("Syntaxe des gabarits 3D", ["scripts/check_templates.py"]),
        ("Langue de l'interface", ["scripts/check_language.py"]),
        ("Tous les etats du tableau de bord", ["scripts/check_app.py"]),
    ]:
        if not run(label, [sys.executable, *command]):
            failures.append(label)

    print(f"\n{BAR}")
    if failures:
        print(f"  {len(failures)} etape(s) en defaut :")
        for item in failures:
            print(f"    - {item}")
        print(f"{BAR}")
        return 1

    print("  Tout est en place.")
    print(f"{BAR}")
    print("\n  Lancer le tableau de bord :\n")
    print("      py -m streamlit run app/main.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
