"""Fotia — tableau de bord des données de combustion en microgravité.

    py -m streamlit run app/main.py

CE QUE CETTE PAGE EST, À CE STADE.

Le squelette de l'architecture décidée : une interface unique, un aiguilleur,
et derrière lui le module compétent pour le régime choisi. Un seul module est
branché sur un modèle entraîné — la suppression, sur PSI-69. Les six autres
sont déclarés dans le registre avec ce qu'ils couvrent, de sorte qu'en ajouter
un revienne à remplir son entrée.

LE PARTI PRIS D'AFFICHAGE.

Une grille de curseurs de 6 400 combinaisons n'en contient que 40 visitées par
un essai réel, soit 0,6 %. Presque partout où l'on peut placer les curseurs,
personne n'a jamais rien mesuré. La probabilité n'est donc JAMAIS affichée
seule : elle vient avec la distance aux données et les essais réels voisins,
et passe en retrait dès que la condition sort du domaine mesuré.

DEUX MODES DE RÉGLAGE, ET POURQUOI.

Annoncer « vous êtes hors du domaine » après coup est nécessaire mais
frustrant : on déplace des curseurs à l'aveugle. Le mode « conditions
testées » remplace donc les curseurs continus par les valeurs réellement
essayées, ce qui rend impossible de tomber entre deux essais sur une variable
donnée.

Cela ne suffit pas, et la carte de couverture explique pourquoi : le domaine
n'est pas une boîte. Choisir une valeur d'oxygène essayée 25 fois et une
valeur de CO2 essayée 12 fois peut donner une CASE que personne n'a visitée.
Le mode « exploration libre » reste disponible pour aller voir ce que le
modèle extrapole, en toute connaissance de cause.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flame.dashboard.coverage import (  # noqa: E402
    coverage_figure,
    coverage_stats,
    nearest_real_test,
    tested_values,
)
from flame.dashboard.proximity import ProximityIndex  # noqa: E402
from flame.dashboard.registry import MODULES, coverage_summary  # noqa: E402

st.set_page_config(page_title="Fotia — combustion en microgravite", layout="wide")

BADGE = {
    "inside": ("#0ca30c", "Dans le domaine teste"),
    "extrapolation": ("#fab219", "Extrapolation"),
    "outside": ("#d03b3b", "Hors du domaine teste"),
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
    return model, X, frame, ProximityIndex(frame, module.features)


def badge(verdict: str) -> None:
    colour, text = BADGE[verdict]
    st.markdown(
        f"<div style='background:{colour};color:#0d0d0d;padding:7px 14px;"
        f"border-radius:6px;display:inline-block;font-weight:700'>{text}</div>",
        unsafe_allow_html=True,
    )


model, X_train, frame, index = load_suppression()
module = MODULES["suppression"]
subset = frame[frame["fuel"] == st.session_state.get("fuel", frame["fuel"].iloc[0])]

st.title("Fotia")
st.caption(
    "Donnees de combustion en microgravite de la NASA, rassemblees et rendues "
    "comparables. 25 investigations, 10 exploitables."
)

tab_predict, tab_coverage, tab_modules, tab_about = st.tabs(
    ["Prediction", "Ou sont les essais", "Couverture des modules", "Methode"]
)

# ---------------------------------------------------------------------------
# Onglet 1 : la prediction, jamais seule
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader(module.name)
    st.caption(f"{module.question}  ·  {module.regime}  ·  {module.investigation}")

    controls, results = st.columns([1, 2], gap="large")

    with controls:
        restricted = st.toggle(
            "Conditions testees seulement",
            value=True,
            help=(
                "Limite chaque curseur aux valeurs reellement essayees par NASA. "
                "Attention : deux valeurs testees separement peuvent former une "
                "combinaison qui ne l'a jamais ete — voir l'onglet « Ou sont les "
                "essais »."
            ),
        )

        fuel = st.selectbox("Carburant", sorted(frame["fuel"].unique()), key="fuel")
        pool = frame[frame["fuel"] == fuel]

        if restricted:
            o2 = st.select_slider(
                "Oxygene", tested_values(pool, "o2_frac"),
                value=min(tested_values(pool, "o2_frac"), key=lambda v: abs(v - 0.21)),
            )
            co2 = st.select_slider("CO2 ajoute", tested_values(pool, "co2_frac"), value=0.0)
            helium = st.select_slider("Helium ajoute", tested_values(pool, "he_frac"), value=0.0)
            pressure_atm = st.slider(
                "Pression (atm)",
                float(pool["pressure_atm"].min()),
                float(pool["pressure_atm"].max()),
                1.0, 0.01,
            )
            d0 = st.slider(
                "Diametre initial (mm)",
                float(pool["d0_mm"].min()), float(pool["d0_mm"].max()),
                float(pool["d0_mm"].median()), 0.01,
            )
            st.caption(
                f"valeurs essayees — O2 : {len(tested_values(pool, 'o2_frac'))}, "
                f"CO2 : {len(tested_values(pool, 'co2_frac'))}, "
                f"He : {len(tested_values(pool, 'he_frac'))}"
            )
        else:
            o2 = st.slider("Oxygene", 0.05, 0.45, 0.21, 0.01)
            co2 = st.slider("CO2 ajoute", 0.0, 0.80, 0.0, 0.01)
            helium = st.slider("Helium ajoute", 0.0, 0.60, 0.0, 0.01)
            pressure_atm = st.slider("Pression (atm)", 0.3, 3.5, 1.0, 0.05)
            d0 = st.slider("Diametre initial (mm)", 0.5, 6.0, 2.5, 0.05)
            st.caption(
                "Mode libre : les curseurs depassent volontairement le domaine "
                "mesure, pour voir ce que le modele extrapole."
            )

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
            opacity = "1" if proximity.trustworthy else ".35"
            st.markdown(
                f"<div style='font-size:52px;font-weight:700;line-height:1;"
                f"opacity:{opacity}'>{probability:.0%}</div>",
                unsafe_allow_html=True,
            )
            if not proximity.trustworthy:
                st.caption("en retrait : la condition sort du domaine mesure")
        with right:
            st.markdown("**Distance aux donnees**")
            badge(proximity.verdict)
            st.caption(proximity.explain())

        for name, (low, high) in proximity.out_of_range.items():
            st.warning(
                f"`{name}` : les essais ne couvrent que {low:g} a {high:g}."
            )

        st.markdown("---")
        st.markdown("**Essais NASA les plus proches**")
        st.caption(
            "De vraies combustions a bord de l'ISS. Elles valent mieux que la "
            "prediction quand elles sont proches."
        )
        columns = [
            "distance", "fuel", "pressure_atm", "o2_frac", "co2_frac",
            "he_frac", "d0_mm", "dext_mm", "test_end",
        ]
        st.dataframe(
            proximity.neighbours[[c for c in columns if c in proximity.neighbours]],
            hide_index=True, use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Onglet 2 : ou sont les essais
# ---------------------------------------------------------------------------
with tab_coverage:
    st.subheader("Le domaine teste n'est pas une boite")
    st.caption(
        "Chaque case est un croisement de deux conditions. Les cases sombres "
        "n'ont jamais ete essayees : ce n'est pas « peu de donnees », c'est aucune."
    )

    axis_options = {
        "oxygene": "o2_frac",
        "CO2 ajoute": "co2_frac",
        "helium ajoute": "he_frac",
    }
    pick = st.columns([1, 1, 2])
    with pick[0]:
        y_name = st.selectbox("Axe vertical", list(axis_options), index=0)
    with pick[1]:
        x_name = st.selectbox("Axe horizontal", list(axis_options), index=1)

    y_key, x_key = axis_options[y_name], axis_options[x_name]
    if x_key == y_key:
        st.info("Choisir deux variables differentes.")
    else:
        stats = coverage_stats(frame, x_key, y_key)
        st.plotly_chart(
            coverage_figure(
                frame, x_key, y_key,
                current=(float(query[x_key]), float(query[y_key])),
            ),
            use_container_width=True,
        )
        st.caption(
            f"Le carre orange marque la position choisie dans l'onglet Prediction. "
            f"{stats['filled']} cases remplies sur {stats['cells']} — le plan "
            "d'experience fait varier un facteur a la fois, il ne balaie pas une grille."
        )

        st.markdown("---")
        st.markdown("**Se placer sur un essai reel**")
        nearest = nearest_real_test(frame, query, module.features)
        display = nearest[
            [c for c in ["fuel", "pressure_atm", "o2_frac", "co2_frac", "he_frac",
                         "d0_mm", "dext_mm", "test_end"] if c in nearest.index]
        ]
        st.dataframe(display.to_frame("valeur").T, hide_index=True,
                     use_container_width=True)
        st.caption(
            "L'essai NASA le plus proche de la position courante. Ses conditions "
            "sont, par construction, dans le domaine teste."
        )

# ---------------------------------------------------------------------------
# Onglet 3 : couverture des modules
# ---------------------------------------------------------------------------
with tab_modules:
    st.subheader("Ce que le tableau de bord sait faire, et ce qui reste a brancher")
    st.caption(
        "Un module ne repond que dans son regime. Chaque jeu de donnees pose une "
        "question differente, avec ses propres entrees et sa propre sortie : il n'y "
        "a pas de modele unique par-dessus, et en fabriquer un serait malhonnete."
    )
    st.dataframe(coverage_summary(), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Onglet 4 : methode
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("Pourquoi la probabilite n'est jamais affichee seule")
    st.markdown(
        """
Sur PSI-69, une grille de curseurs a dix crans sur l'oxygene, dix sur le CO2,
huit sur l'helium et huit sur la pression ouvre **6 400 combinaisons**. Quarante
d'entre elles ont ete visitees par au moins un essai reel, soit **0,6 %**.

Le plan d'experience fait varier **un facteur a la fois**. Sur le croisement
oxygene x CO2, la colonne sans CO2 est complete sur les 14 niveaux d'oxygene,
puis chaque niveau d'oxygene n'a ete croise qu'avec un ou deux niveaux de CO2 :
31 cases remplies sur 196. D'ou un piege contre-intuitif — **choisir deux
valeurs testees separement ne garantit pas que leur combinaison l'ait ete.**

Exemple trouve par le voyant lui-meme : FLEX-1 n'a **jamais** combine CO2 et
helium. Zero essai sur 213. Le tableau de bord ne peut donc rien dire de leur
effet conjoint, et il le signale au lieu de repondre quand meme.

Le seuil qui separe « dans le domaine » de « extrapolation » n'est pas choisi a
la main. On mesure d'abord, a l'interieur du jeu de donnees, la distance de
chaque essai a son plus proche voisin : cette distribution dit ce qu'est un
voisinage normal pour ces donnees-la. Le seuil se recalibre donc tout seul pour
un autre module.

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
