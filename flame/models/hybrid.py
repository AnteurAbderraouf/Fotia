"""Moteur hybride de la flamme — appris dans les donnees, physique au-dela.

LE PRINCIPE (option B, choisie le 2026-11-09).

    dans le domaine teste   les modeles PSI-69 tels quels : K (loi en d2),
                            probabilite d'extinction, diametre d'extinction.
    hors du domaine         K = K appris AU BORD du domaine x rapport
                            physique ln(1+B) entre le bord et la demande.
                            L'extinction n'est PAS extrapolee : on affiche la
                            valeur au bord, nommee comme telle.

Rien n'est recalcule a cote des modeles existants : `flame/models/` et
`flame/dashboard/predict.py` sont appeles, pas reecrits.

LA PHYSIQUE, ET SES LIMITES MESUREES SUR NOS PROPRES DONNEES.

Theorie quasi-stationnaire de Spalding/Godsave :

    K = 8 (lambda_g / cp_g) / rho_l  x  ln(1 + B)
    B = Y_O2 x PCI / (nu x L_v)

Les proprietes viennent de data/context/proprietes_combustibles.csv (NIST,
PubChem). lambda_g / cp_g est suppose identique entre deux etats compares, ce
qui permet de ne jamais en avoir besoin : on ne calcule que des RAPPORTS.

Confrontee a PSI-69 avant d'ecrire ce fichier :

    effet de l'O2, methanol 12 -> 30 %   physique x1.66   mesure x2.09
    effet de l'O2, heptane  14 -> 30 %   physique x1.34   mesure x1.74
    heptane / methanol a 21 %            physique  2.02   mesure  1.06

Deux consequences, ecrites a l'ecran par le moteur lui-meme :
  * vers plus d'oxygene, la physique sous-estime K d'environ 20-25 % ;
  * d'un carburant a l'autre, elle peut se tromper d'un facteur 2. Le
    n-decane (aucun essai PSI-69) est calcule depuis l'heptane, deux alcanes
    voisins ; c'est une hypothese, testee a l'onglet de validation.

Terme sensible cp (T_inf - T_eb) neglige dans B : il pese quelques % de B
pour ces carburants, et son cp n'est pas dans nos sources.

La pression n'entre pas dans B : au premier ordre, la theorie quasi-
stationnaire ne donne pas d'effet de pression sur K. Hors de la plage de
pression testee, K reste donc celui du bord, et le moteur le dit.

LA GRAVITE. Aucune table entrainable n'a d'essai hors microgravite. Les
chiffres restent ceux de la microgravite ; seule la forme dessinee change,
et elle est etiquetee illustrative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from flame.dashboard.predict import (
    Bundle,
    predicted_diameter,
    predicted_rate,
    probability,
)
from flame.loaders.context import constant, fuel_table, load_fuel_properties
from flame.models.droplet_burn import burn_duration, rate_frame

MMHG_PER_ATM = 760.0
MICROGRAVITY_BELOW = 0.01

# Carburant affiche -> cle de proprietes_combustibles.csv
FUEL_KEYS = {"Methanol": "methanol", "Heptane": "n-heptane", "n-decane": "n-decane"}
# Carburant sans essai -> carburant appris qui lui sert de point d'appui
SURROGATE = {"n-decane": "Heptane"}
FUELS = list(FUEL_KEYS)

NUMERIC = ["pressure_atm", "o2_frac", "co2_frac", "he_frac", "d0_mm"]
LABELS = {
    "pressure_atm": "pressure",
    "o2_frac": "oxygen",
    "co2_frac": "added CO2",
    "he_frac": "added helium",
    "d0_mm": "droplet size",
    "gravity_g": "gravity",
}


def _atoms(formula: str) -> tuple[int, int, int]:
    counts = {e: int(n or 1) for e, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula)}
    return counts.get("C", 0), counts.get("H", 0), counts.get("O", 0)


class Physics:
    """Le nombre de transfert B et le facteur ln(1+B)/rho_l, par carburant."""

    def __init__(self, raw: pd.DataFrame | None = None):
        raw = load_fuel_properties() if raw is None else raw
        self.fuels = fuel_table(raw)
        # Enthalpie de vaporisation de l'eau = difference des deux enthalpies
        # de formation CODATA ; sert a passer du PCS (eau liquide) au PCI.
        self.water_vap = constant(raw, "eau", "enthalpie_formation_gaz") - constant(
            raw, "eau", "enthalpie_formation_liquide"
        )
        self.molar_mass = {
            "o2": constant(raw, "oxygene", "masse_molaire"),
            "n2": constant(raw, "azote", "masse_molaire"),
            "co2": constant(raw, "dioxyde_de_carbone", "masse_molaire"),
            "he": constant(raw, "helium", "masse_molaire"),
        }

    def heating_value(self, fuel_key: str) -> float:
        """PCI en kJ/g."""
        row = self.fuels.loc[fuel_key]
        _, h, _ = _atoms(row["formule"])
        hhv = -row["enthalpie_combustion_liquide"]
        return (hhv - h / 2 * self.water_vap) / row["masse_molaire"]

    def stoichiometric_ratio(self, fuel_key: str) -> float:
        """Masse d'O2 par masse de carburant."""
        row = self.fuels.loc[fuel_key]
        c, h, o = _atoms(row["formule"])
        return (c + h / 4 - o / 2) * self.molar_mass["o2"] / row["masse_molaire"]

    def oxygen_mass_fraction(self, o2: float, co2: float, he: float) -> float:
        n2 = 1.0 - o2 - co2 - he
        if n2 < -1e-9:
            raise ValueError("oxygen plus diluents exceed 100 % of the gas")
        m = self.molar_mass
        total = o2 * m["o2"] + max(n2, 0.0) * m["n2"] + co2 * m["co2"] + he * m["he"]
        return o2 * m["o2"] / total

    def transfer_number(self, fuel_key: str, o2: float, co2: float, he: float) -> float:
        row = self.fuels.loc[fuel_key]
        latent = row["enthalpie_vaporisation"] / row["masse_molaire"]
        y = self.oxygen_mass_fraction(o2, co2, he)
        return y * self.heating_value(fuel_key) / (self.stoichiometric_ratio(fuel_key) * latent)

    def factor(self, fuel_key: str, o2: float, co2: float, he: float) -> float:
        """ln(1+B)/rho_l : K a une constante de transport pres."""
        b = self.transfer_number(fuel_key, o2, co2, he)
        return float(np.log1p(b) / self.fuels.loc[fuel_key, "masse_volumique_liquide"])


@dataclass
class Prediction:
    fuel: str
    k_mm2_s: float
    k_basis: str  # learned | physics | surrogate
    extinction_probability: float | None
    extinction_basis: str  # learned | edge | none
    dext_mm: float | None
    duration_s: float
    extinguishes: bool | None
    verdict: str  # inside | extrapolation | outside
    extrapolated: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


class Engine:
    def __init__(self, bundle: Bundle, physics: Physics | None = None):
        if bundle.module.key != "suppression":
            # Message en anglais : une exception peut remonter jusqu'a la page.
            raise ValueError("the flame engine needs the Suppression module (PSI-69)")
        self.bundle = bundle
        self.physics = physics or Physics()
        # DEUX PLAGES TESTEES, UNE PAR REPONSE (2026-11-09).
        #   ranges      ce qui a ete BRULE et dont K est mesure : FLEX + MSL-1.
        #               C'est la plage « Tested » des curseurs.
        #   ext_ranges  ce dont l'issue (extinction ou non) est connue : FLEX
        #               seul, essais etiquetes. Plus etroite.
        # Avant, une seule plage (la seconde) servait aux deux : trois
        # combustions d'heptane a 34 % d'O2, mesurees mais sans etiquette,
        # et tout MSL-1 restaient « jamais testes ».
        # Plages PAR CARBURANT : l'heptane n'a pas ete brule partout ou le
        # methanol l'a ete.
        def per_fuel(frame):
            frame = frame.assign(pressure_atm=frame["pressure_mmhg"] / MMHG_PER_ATM)
            return {
                fuel: {
                    name: (float(block[name].min()), float(block[name].max()))
                    for name in NUMERIC
                }
                for fuel, block in frame.groupby("fuel")
            }

        self.ext_ranges = per_fuel(bundle.frame)
        self.ranges = per_fuel(rate_frame())

    @staticmethod
    def _features(fuel: str, values: dict) -> dict:
        return {
            "fuel": fuel,
            "pressure_mmhg": values["pressure_atm"] * MMHG_PER_ATM,
            "o2_frac": values["o2_frac"],
            "co2_frac": values["co2_frac"],
            "he_frac": values["he_frac"],
            "d0_mm": values["d0_mm"],
        }

    def predict(self, conditions: dict) -> Prediction:
        fuel = conditions["fuel"]
        model_fuel = SURROGATE.get(fuel, fuel)
        ranges = self.ranges[model_fuel]
        values = {name: float(conditions[name]) for name in NUMERIC}
        gravity = float(conditions.get("gravity_g", 0.0))

        def beyond(box):
            return {
                name: (lo, hi, values[name])
                for name, (lo, hi) in box.items()
                if not lo - 1e-9 <= values[name] <= hi + 1e-9
            }

        # Au-dela de tout essai brule : c'est ce que l'ecran appelle « beyond
        # the tests ».
        extrapolated = beyond(ranges)
        anchor = {name: float(np.clip(values[name], *ranges[name])) for name in NUMERIC}
        # Au-dela des seuls essais dont l'issue est connue.
        ext_box = self.ext_ranges[model_fuel]
        ext_beyond = beyond(ext_box)
        ext_anchor = {name: float(np.clip(values[name], *ext_box[name])) for name in NUMERIC}
        notes: list[str] = []

        # --- K : appris au bord, puis rapport physique jusqu'a la demande ---
        k_anchor = predicted_rate(self.bundle, self._features(model_fuel, anchor))
        physics_ratio = self.physics.factor(
            FUEL_KEYS[fuel], values["o2_frac"], values["co2_frac"], values["he_frac"]
        ) / self.physics.factor(
            FUEL_KEYS[model_fuel], anchor["o2_frac"], anchor["co2_frac"], anchor["he_frac"]
        )
        k = max(k_anchor * physics_ratio, 1e-3)

        if fuel in SURROGATE:
            k_basis = "surrogate"
            notes.append(
                f"No NASA test burned {fuel} in this dataset. Its burning rate is "
                f"computed from {model_fuel.lower()} with combustion theory; "
                "between fuels that theory was off by up to a factor of 2 on our "
                "own data."
            )
        elif extrapolated:
            k_basis = "physics"
        else:
            k_basis = "learned"

        if "o2_frac" in extrapolated and values["o2_frac"] > ranges["o2_frac"][1]:
            notes.append(
                "Above the oxygen ever tested, the burning rate comes from "
                "combustion theory. On the tested range that theory "
                "under-predicts the effect of oxygen by about 20-25 %, so the "
                "real rate is probably higher."
            )
        if "pressure_atm" in extrapolated:
            notes.append(
                "Pressure is outside the tests. Theory gives no first-order "
                "pressure effect on the burning rate, so it is held at the "
                "nearest tested pressure."
            )
        if "co2_frac" in extrapolated or "he_frac" in extrapolated:
            notes.append(
                "Diluent beyond the tests: only its effect on the oxygen mass "
                "fraction is accounted for, not its heat transport."
            )
        if "d0_mm" in extrapolated:
            notes.append(
                "Droplet size outside the tests: the burning rate and extinction "
                "are taken at the nearest tested size."
            )

        # --- extinction : apprise dans le domaine, jamais extrapolee ---
        if fuel in SURROGATE:
            p, basis, dext = None, "none", None
            notes.append(f"Whether a {fuel} flame goes out is not modelled: no test.")
        else:
            point = values if not ext_beyond else ext_anchor
            features = self._features(fuel, point)
            p = probability(self.bundle, features)
            dext = predicted_diameter(self.bundle, features)
            basis = "learned" if not ext_beyond else "edge"
            if ext_beyond and not extrapolated:
                notes.append(
                    "NASA burned droplets at this setting and measured how fast "
                    "they burn, but no test here records whether the flame went "
                    "out. The chance of extinction is the value at the edge of "
                    "the tests that do."
                )
            elif ext_beyond:
                notes.append(
                    "The chance of extinction shown is the value at the edge of "
                    "the tests, not a prediction at this condition."
                )

        extinguishes = None if p is None else p >= 0.5
        final = dext if (extinguishes and dext is not None) else 0.0
        final = float(np.clip(final, 0.0, values["d0_mm"]))
        duration = burn_duration(values["d0_mm"], final, k)

        # --- badge ---
        if gravity >= MICROGRAVITY_BELOW:
            extrapolated["gravity_g"] = (0.0, 0.0, gravity)
            notes.append(
                "Every test behind these numbers ran in microgravity. With "
                "gravity, buoyancy reshapes the flame; the numbers do not "
                "include it, and the flame shape drawn is illustrative."
            )
        if extrapolated or fuel in SURROGATE:
            verdict = "outside"
        elif ext_beyond:
            # Brule et mesure ici, issue jamais observee : entre les essais.
            verdict = "extrapolation"
        else:
            query = self._features(fuel, values)
            verdict = self.bundle.index.assess(query).verdict

        return Prediction(
            fuel=fuel,
            k_mm2_s=k,
            k_basis=k_basis,
            extinction_probability=p,
            extinction_basis=basis,
            dext_mm=dext,
            duration_s=duration,
            extinguishes=extinguishes,
            verdict=verdict,
            extrapolated=extrapolated,
            notes=notes,
            ranges=ranges,
        )


def main() -> None:
    from flame.dashboard.predict import build
    from flame.dashboard.registry import MODULES

    engine = Engine(build(MODULES["suppression"]))
    physics = engine.physics
    for key in FUEL_KEYS.values():
        print(f"{key:10} PCI {physics.heating_value(key) * 1000:6.0f} kJ/kg  "
              f"B(air) {physics.transfer_number(key, 0.21, 0, 0):5.2f}")
    base = {"pressure_atm": 1.0, "o2_frac": 0.21, "co2_frac": 0.0, "he_frac": 0.0,
            "d0_mm": 2.5, "gravity_g": 0.0}
    cases = [
        ("ISS, methanol", {**base, "fuel": "Methanol"}),
        ("ISS, heptane", {**base, "fuel": "Heptane"}),
        ("ISS, n-decane", {**base, "fuel": "n-decane"}),
        ("Apollo flight, heptane", {**base, "fuel": "Heptane", "pressure_atm": 0.34, "o2_frac": 1.0}),
        ("Apollo 1 pad, methanol", {**base, "fuel": "Methanol", "pressure_atm": 1.136, "o2_frac": 1.0, "gravity_g": 1.0}),
        ("Mars, heptane", {**base, "fuel": "Heptane", "pressure_atm": 0.558, "o2_frac": 0.34, "gravity_g": 0.379}),
    ]
    for name, c in cases:
        r = engine.predict(c)
        p = "  -  " if r.extinction_probability is None else f"{r.extinction_probability:.2f}"
        print(f"{name:24} K {r.k_mm2_s:.3f} ({r.k_basis:9}) P(ext) {p} ({r.extinction_basis:7}) "
              f"t {r.duration_s:5.1f}s  {r.verdict:13} hors: {list(r.extrapolated)}")


if __name__ == "__main__":
    main()
