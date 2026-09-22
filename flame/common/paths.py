"""Chemins du projet.

Règle fondatrice : `combustion_science/` est une source de vérité en LECTURE
SEULE. Aucun code de ce projet n'y écrit jamais. Tout ce qui est généré va
dans `data/processed/` et doit pouvoir être reconstruit de zéro — si tu
supprimes ce dossier, un re-run des loaders le régénère à l'identique.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Source de vérité NASA — lecture seule, jamais modifiée
NASA_ROOT = PROJECT_ROOT / "combustion_science"
MICROGRAVITY = NASA_ROOT / "microgravity_investigation"
GROUND = NASA_ROOT / "ground_investigation"

# Sorties générées — jetables et reconstructibles
PROCESSED = PROJECT_ROOT / "data" / "processed"


def psi_dir(psi_id: str) -> Path:
    """Dossier d'une investigation, sans avoir à se rappeler sa catégorie NASA.

    PSI-117 par exemple est rangé sous ground_investigation alors que NASA le
    décrit comme une investigation en vol (cf. handoff §6) — cette fonction
    évite d'avoir à s'en souvenir à chaque appel.
    """
    for parent in (MICROGRAVITY, GROUND):
        candidate = parent / psi_id
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"{psi_id} introuvable sous {MICROGRAVITY} ni {GROUND}"
    )


def processed_path(name: str) -> Path:
    """Chemin d'un fichier généré, en créant le dossier au besoin."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    return PROCESSED / name
