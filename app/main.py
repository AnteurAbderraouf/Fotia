"""Fotia — tableau de bord des données de combustion en microgravité.

    py -m streamlit run app/main.py

CE QUE CETTE PAGE EST, À CE STADE.

Le squelette de l'architecture décidée : une interface unique, un aiguilleur,
et derrière lui le module compétent pour le régime choisi. Un seul module est
branché sur un modèle entraîné — la suppression, sur PSI-69. Les six autres
sont déclarés dans le registre avec ce qu'ils couvrent, de sorte qu'en ajouter
un revienne à remplir son entrée.

LE PARTI PRIS D'AFFICHAGE.

Une mesure faite sur PSI-69 : une grille de curseurs de 6 400 combinaisons
n'en contient que 40 visitées par un essai réel, soit 0,6 %. Presque partout
où l'on peut placer les curseurs, personne n'a jamais rien mesuré.

La probabilité n'est donc JAMAIS affichée seule. Elle vient toujours avec la
distance aux données et les essais réels voisins. Quand la condition sort du
domaine testé, le chiffre est affiché en retrait, parce qu'il ne vaut plus
grand-chose — et ce que NASA n'a pas testé est, pour un ingénieur sécurité,
une information au moins aussi utile que ce qu'elle a testé.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flame.dashboard.proximity import ProximityIndex  # noqa: E402
from flame.dashboard.registry import MODULES, coverage_summary  # noqa: E402

st.set_page_config(page_title="Fotia — combustion en microgravite", layout="wide")

BADGE = {
    "inside": ("#1f7a4d", "Dans le domaine teste"),
    "extrapolation": ("#8a6d1f", "Extrapolation"),
    "outside": ("#8a2f2f", "Hors du domaine teste"),
}


@st.cache_resource
def load_suppression():
    """Charge les données et entraîne le modèle une fois pour toutes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from flame.models.suppression import prepare

    X, y = prepare()
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000, class_weight="balanced", random_state=0
                ),
            ),
        ]
    ).fit(X, y)

    module = MODULES["suppression"]
    frame = module.loader()
    frame = frame[frame[module.features].notna().all(axis=1)].reset_index(drop=True)
    index = ProximityIndex(frame, module.features)
    return model, X, frame, index


def badge(verdict: str) -> None:
    colour, text = BADGE[verdict]
    st.markdown(
        f"<div style='background:{colour};color:#fff;padding:7px 14px;"
        f"border-radius:6px;display:inline-block;font-weight:600'>{text}</div>",
        unsafe_allow_html=True,
    )


st.title("Fotia")
st.caption(
    "Donnees de combustion en microgravite de la NASA, rassemblees et rendues "
    "comparables. 25 investigations, 10 exploitables."
)

tab_predict, tab_catalogue, tab_about = st.tabs(
    ["Prediction", "Couverture des modules", "Methode"]
)

with tab_predict:
    model, X_train, frame, index = load_suppression()
    module = MODULES["suppression"]

    st.subheader(module.name)
    st.caption(f"{module.question}  ·  {module.regime}  ·  {module.investigation}")

    controls, results = st.columns([1, 2], gap="large")

    with controls:
        st.markdown("**Conditions**")
        fuel = st.selectbox("Carburant", sorted(frame["fuel"].unique()))
        o2 = st.slider("Oxygene (fraction molaire)", 0.05, 0.45, 0.21, 0.01)
        co2 = st.slider("CO2 ajoute (fraction molaire)", 0.0, 0.80, 0.0, 0.01)
        helium = st.slider("Helium ajoute (fraction molaire)", 0.0, 0.60, 0.0, 0.01)
        pressure_atm = st.slider("Pression (atm)", 0.3, 3.5, 1.0, 0.05)
        d0 = st.slider("Diametre initial de la goutte (mm)", 0.5, 6.0, 2.5, 0.05)

        total = o2 + co2 + helium
        if total > 1.0:
            st.error(
                f"Les fractions molaires somment a {total:.2f}. "
                "Un melange ne peut pas depasser 1."
            )
            st.stop()
        st.caption(f"azote de complement : {1 - total:.2f}")

    query = {
        "fuel": fuel,
        "pressure_mmhg": pressure_atm * 760.0,
        "o2_frac": o2,
        "co2_frac": co2,
        "he_frac": helium,
        "d0_mm": d0,
    }

    row = pd.DataFrame([query])
    row["fuel_methanol"] = (row.pop("fuel") == "Methanol").astype(int)
    probability = float(model.predict_proba(row[X_train.columns])[0, 1])

    proximity = index.assess(query)

    with results:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Probabilite d'extinction**")
            if proximity.trustworthy:
                st.markdown(
                    f"<div style='font-size:52px;font-weight:700;line-height:1'>"
                    f"{probability:.0%}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='font-size:52px;font-weight:700;line-height:1;"
                    f"opacity:.35'>{probability:.0%}</div>",
                    unsafe_allow_html=True,
                )
                st.caption("affichee en retrait : la condition sort du domaine mesure")
        with right:
            st.markdown("**Distance aux donnees**")
            badge(proximity.verdict)
            st.caption(proximity.explain())

        if proximity.out_of_range:
            for name, (low, high) in proximity.out_of_range.items():
                st.warning(
                    f"`{name}` : aucun essai entre {low:g} et {high:g} ne couvre "
                    "la valeur demandee."
                )

        st.markdown("---")
        st.markdown("**Essais NASA les plus proches**")
        st.caption(
            "Ce sont de vraies combustions a bord de l'ISS. Elles valent mieux "
            "que la prediction quand elles sont proches."
        )
        columns = [
            "distance",
            "fuel",
            "pressure_atm",
            "o2_frac",
            "co2_frac",
            "he_frac",
            "d0_mm",
            "dext_mm",
            "test_end",
        ]
        st.dataframe(
            proximity.neighbours[[c for c in columns if c in proximity.neighbours]],
            hide_index=True,
            use_container_width=True,
        )

with tab_catalogue:
    st.subheader("Ce que le tableau de bord sait faire, et ce qui reste a brancher")
    st.caption(
        "Un module ne repond que dans son regime. Chaque jeu de donnees pose une "
        "question differente, avec ses propres entrees et sa propre sortie : il n'y "
        "a pas de modele unique par-dessus, et en fabriquer un serait malhonnete."
    )
    st.dataframe(coverage_summary(), hide_index=True, use_container_width=True)

with tab_about:
    st.subheader("Pourquoi la probabilite n'est jamais affichee seule")
    st.markdown(
        """
Sur PSI-69, une grille de curseurs a dix crans sur l'oxygene, dix sur le CO2,
huit sur l'helium et huit sur la pression ouvre **6 400 combinaisons**. Quarante
d'entre elles ont ete visitees par au moins un essai reel, soit **0,6 %**.

Presque partout ou l'on peut placer les curseurs, personne n'a jamais rien
mesure. Un tableau de bord qui repondrait une probabilite du meme air assure a
chacune de ces positions mentirait par omission dans la quasi-totalite des cas.

Le seuil qui separe « dans le domaine » de « extrapolation » n'est pas choisi a
la main. On mesure d'abord, a l'interieur du jeu de donnees, la distance de
chaque essai a son plus proche voisin : cette distribution dit ce qu'est un
voisinage normal pour ces donnees-la. Une condition demandee est ensuite jugee
par rapport a cette reference, et le seuil se recalibre tout seul pour un autre
module.

**Ce que NASA n'a pas teste est une information.** Pour un ingenieur securite
incendie, apprendre qu'aucune campagne n'a explore son materiau a 30 %
d'oxygene vaut au moins autant qu'une probabilite.
        """
    )
    st.markdown("---")
    st.caption(
        "Les features n'utilisent que ce qui est connu AVANT l'allumage. Le diametre "
        "d'extinction, la duree de combustion et le taux de combustion sont mesures "
        "pendant ou apres : les donner en entree reviendrait a predire l'extinction "
        "a partir de la preuve qu'elle a eu lieu."
    )
