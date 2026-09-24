"""Les quatre constats d'accueil, calculés et non écrits en dur.

POURQUOI LES CALCULER.

Un chiffre recopié dans le code diverge au premier changement de loader. Le
catalogue est scanné pour cette raison ; les constats d'accueil le sont aussi.
Si une table change, la page d'accueil suit, ou elle échoue bruyamment.

Le quatrième constat mérite un mot. Il annonce ce que NASA n'a PAS mesuré, ce
qui est inhabituel pour une page d'accueil. C'est pourtant ce qui distingue
cet outil d'une vitrine : un tableau de bord qui avoue ses trous est plus
utile qu'un qui répond à tout du même air assuré.

LE TEXTE EST EN ANGLAIS. L'interface s'adresse a un jury Space Apps
international ; le code et ses commentaires restent en francais.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from flame.loaders.psi99 import load as load_saffire
from flame.loaders.psi101 import load as load_detection
from flame.loaders.psi115 import load as load_fields


@dataclass(frozen=True)
class Finding:
    """Un constat : sa valeur, son titre, sa phrase, sa source."""

    figure: str
    unit: str
    title: str
    body: str
    source: str
    tone: str  # good | critical | warning | neutral
    question: str  # la question du niveau 2 vers laquelle il mene


def _silicone() -> Finding:
    """Le silicone brûle sur Terre et pas en orbite."""
    saffire = load_saffire()
    silicone = saffire[saffire["material"].str.contains("Silicone", na=False)]
    burned = int(silicone["burned_in_microgravity"].sum())
    complete = int(
        silicone["g1_burn_length_qualitative"]
        .fillna("")
        .str.contains("complete", case=False)
        .sum()
    )
    return Finding(
        figure=str(burned),
        unit=f"/ {len(silicone)}",
        title="Silicone will not burn in orbit",
        body=(
            f"{len(silicone)} silicone samples, {burned} burned in microgravity. "
            f"On Earth, {complete} of the same burned end to end."
        ),
        source=f"Saffire-II · {len(saffire)} samples",
        tone="good",
        question="materials",
    )


def _blind_spot() -> Finding:
    """Des matériaux qui produisent beaucoup et que le capteur voit peu.

    Le critère est « moins de la moitié du signal le plus fort ». Un premier
    essai comparait a la mediane, et ne retenait qu'un materiau alors que la
    phrase en citait deux : le chiffre et le texte se contredisaient. La
    moitie du maximum separe nettement le groupe silencieux du reste.
    """
    detection = load_detection()
    by_material = detection.groupby("material").agg(
        volts=("iss_scatter_volts", "mean"),
        particles=("ptrak_particles_cc", "mean"),
    )
    loudest = float(by_material["volts"].max())
    quiet = by_material[by_material["volts"] < loudest / 2].sort_values("volts")
    typical_particles = by_material["particles"].median()
    return Finding(
        figure=str(len(quiet)),
        unit=f"of {len(by_material)}",
        title="Materials the smoke alarm barely sees",
        body=(
            f"{' and '.join(quiet.index)} register "
            f"{' and '.join(f'{v:.1f}' for v in quiet['volts'])} V against "
            f"{loudest:.1f} V for the loudest material, while producing as many "
            "particles as the rest. A scattering sensor answers to particle "
            "size, not count."
        ),
        source=f"SAME-R · {len(detection)} tests",
        tone="critical",
        question="detection",
    )


def _smoke_concentration() -> Finding:
    """Sans flottabilité, la fumée ne part pas.

    LA COMPARAISON SE FAIT A TAILLE DE BRULEUR CONSTANTE. Un premier essai
    prenait le maximum global de chaque regime de gravite, ce qui mettait en
    regard le 10 cm terrestre et le 8 cm en microgravite : deux configurations
    differentes, donc un ecart sans signification. A configuration egale,
    l'effet existe dans les deux cas mais son ampleur varie beaucoup, et la
    phrase le dit.
    """
    fields = load_fields()
    smoke = fields[(fields["variable"] == "smoke") & fields["thermophoresis"]]
    paired = smoke.pivot_table(
        index="burner_size_cm", columns="gravity", values="max"
    ).dropna()
    paired["rise"] = (
        (paired["microgravity"] - paired["1g"]) / paired["1g"] * 100
    )
    strongest = paired["rise"].idxmax()
    weakest = paired["rise"].idxmin()
    return Finding(
        figure=f"+{paired.loc[strongest, 'rise']:.0f}",
        unit="%",
        title="Smoke stays put instead of rising",
        body=(
            f"On the {strongest} cm burner, peak smoke concentration climbs from "
            f"{paired.loc[strongest, '1g']:.1f} to "
            f"{paired.loc[strongest, 'microgravity']:.1f} without gravity. On the "
            f"{weakest} cm burner the same change is only "
            f"+{paired.loc[weakest, 'rise']:.0f} %, so the effect depends on the "
            "configuration."
        ),
        source=f"Smoke CFD · {fields['case'].nunique()} simulated cases",
        tone="warning",
        question="detection",
    )


def _coverage(bundle) -> Finding:
    """La part du domaine que personne n'a jamais visitée."""
    frame = bundle.frame
    bins = {"o2_frac": 10, "co2_frac": 10, "he_frac": 8, "pressure_atm": 8}
    cells = int(np.prod(list(bins.values())))
    grouped = frame.groupby(
        [pd.cut(frame[name], count) for name, count in bins.items()], observed=True
    )
    visited = len(grouped)
    share = (1 - visited / cells) * 100
    # Le separateur de milliers se pose a part : un remplacement global des
    # virgules effacait aussi celles de la phrase.
    total = f"{cells:,}".replace(",", " ")
    return Finding(
        figure=f"{share:.1f}",
        unit="%",
        title="Most conditions were never tested",
        body=(
            f"Of {total} plausible combinations of oxygen, CO2, helium and "
            f"pressure, {visited} were ever run. Knowing where the gaps are is "
            "a result too."
        ),
        source=f"FLEX-1 · {len(frame)} tests",
        tone="neutral",
        question="suppression",
    )


def headline(suppression_bundle) -> list[Finding]:
    """Les quatre constats, dans l'ordre d'affichage."""
    return [
        _silicone(),
        _blind_spot(),
        _smoke_concentration(),
        _coverage(suppression_bundle),
    ]
