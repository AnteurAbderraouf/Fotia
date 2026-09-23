"""Fotia — tableau de bord des données de combustion en microgravité.

    py -m streamlit run app/main.py

L'ARCHITECTURE. Une interface unique, un aiguilleur, et derrière lui le module
compétent pour le régime choisi. Le §8 du handoff tient toujours — pas de
modèle unique sur toutes les données, les jeux ne partagent ni entrées ni
sorties — mais une interface unifiée n'est pas un modèle unifié.

La page ne code aucun module en dur : elle dessine les commandes déclarées
dans le registre. En brancher un de plus revient à remplir son entrée.

LE PARTI PRIS D'AFFICHAGE. Une grille de curseurs de 6 400 combinaisons n'en
contient que 40 visitées par un essai réel, soit 0,6 %. La probabilité n'est
donc JAMAIS affichée seule : elle vient avec la distance aux données et les
essais réels voisins, et passe en retrait dès que la condition sort du domaine
mesuré.

DEUX MODES DE RÉGLAGE. Le mode « conditions testées » limite chaque commande
aux valeurs réellement essayées, ce qui empêche de tomber entre deux essais.
Cela ne suffit pas : le domaine n'est pas une boîte, et deux valeurs testées
séparément peuvent former une combinaison qui ne l'a jamais été. L'onglet de
couverture est là pour ça.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flame.dashboard.catalogue import (  # noqa: E402
    VERDICT_LABELS,
    read_info_text,
    scan,
    summary,
    table,
)
from flame.dashboard.coverage import (  # noqa: E402
    coverage_figure,
    coverage_stats,
    nearest_real_test,
    tested_values,
)
from flame.dashboard.predict import (  # noqa: E402
    build,
    predicted_diameter,
    predicted_rate,
    probability,
    to_features,
)
from flame.dashboard.registry import (  # noqa: E402
    MODULES,
    coverage_summary,
    ready_modules,
)
from flame.models.droplet_burn import burn_duration  # noqa: E402
from flame.viz.droplet_anim import page as droplet_page  # noqa: E402
from flame.viz.droplet_live import page as live_page  # noqa: E402

st.set_page_config(page_title="Fotia — combustion en microgravite", layout="wide")

BADGE = {
    "inside": ("#0ca30c", "Dans le domaine teste"),
    "extrapolation": ("#fab219", "Extrapolation"),
    "outside": ("#d03b3b", "Hors du domaine teste"),
}

OUTCOME_LABELS = {
    "suppression": ("Probabilite d'extinction", "la flamme s'eteint"),
    "sustainment": ("Probabilite d'auto-extinction", "la flamme meurt seule"),
}


@st.cache_resource
def bundle_for(key: str):
    return build(MODULES[key])


def badge(verdict: str) -> None:
    colour, text = BADGE[verdict]
    st.markdown(
        f"<div style='background:{colour};color:#0d0d0d;padding:7px 14px;"
        f"border-radius:6px;display:inline-block;font-weight:700'>{text}</div>",
        unsafe_allow_html=True,
    )


def render_control(control, pool: pd.DataFrame, restricted: bool):
    """Dessine une commande à partir de sa déclaration dans le registre."""
    series = pool[control.feature]
    caption = f"{control.label}" + (f" ({control.unit})" if control.unit else "")
    hint = control.help or None

    if not pd.api.types.is_numeric_dtype(series):
        return st.selectbox(caption, sorted(series.dropna().unique()), help=hint)

    values = tested_values(pool, control.feature)
    low, high = float(series.min()), float(series.max())

    if restricted and len(values) <= 20:
        middle = values[len(values) // 2]
        return st.select_slider(caption, values, value=middle, help=hint)
    if restricted:
        return st.slider(caption, low, high, float(series.median()), help=hint)

    margin = (high - low) * 0.35 or 1.0
    return st.slider(
        caption, low - margin, high + margin, float(series.median()), help=hint
    )


st.title("Fotia")
st.caption(
    "Donnees de combustion en microgravite de la NASA, rassemblees et rendues "
    "comparables. 24 investigations, 10 exploitables, 2 modules branches."
)

available = ready_modules()
choice = st.radio(
    "Regime",
    list(available),
    format_func=lambda key: f"{available[key].name}  ·  {available[key].investigation}",
    horizontal=True,
    label_visibility="collapsed",
)
module = available[choice]
data = bundle_for(choice)

st.caption(f"**{module.question}**  ·  {module.regime}")
if module.note:
    st.caption(module.note)

tabs = ["Prediction", "Ou sont les essais", "Le catalogue", "Modules", "Methode"]
if choice == "suppression":
    tabs.insert(0, "La flamme, en direct")
rendered = st.tabs(tabs)
if choice == "suppression":
    (tab_live, tab_predict, tab_coverage, tab_catalogue, tab_modules,
     tab_about) = rendered
else:
    tab_live = None
    (tab_predict, tab_coverage, tab_catalogue, tab_modules,
     tab_about) = rendered

# ---------------------------------------------------------------------------
# Onglet interactif : tout se calcule dans le navigateur, donc sans coupure.
# ---------------------------------------------------------------------------
if tab_live is not None:
    with tab_live:
        st.caption(
            "Les curseurs sont **dans la vue**. Streamlit relance tout le script a "
            "chaque reglage, ce qui recreait l'iframe et produisait un blanc puis un "
            "saut ; ici les trois modeles — tous lineaires, donc reductibles a un "
            "produit scalaire — sont calcules par la page elle-meme. Rien ne repasse "
            "par le serveur, et la gouttelette glisse d'un etat a l'autre."
        )
        components.html(live_page(data), height=560, scrolling=False)
        st.caption(
            "Le retrecissement suit la **loi en d²** : d²(t) = d₀² − K·t, verifiee "
            "sur les durees mesurees a 0.995 de correlation. Le voyant de distance "
            "et les essais voisins sont recalcules dans la page a partir des 206 "
            "essais embarques : mêmes seuils, mêmes donnees qu'en Python."
        )

# ---------------------------------------------------------------------------
with tab_predict:
    controls, results = st.columns([1, 2], gap="large")

    with controls:
        restricted = st.toggle(
            "Conditions testees seulement",
            value=True,
            key=f"restrict_{choice}",
            help=(
                "Limite chaque commande aux valeurs reellement essayees. "
                "Attention : deux valeurs testees separement peuvent former une "
                "combinaison qui ne l'a jamais ete."
            ),
        )

        query: dict = {}
        pool = data.frame
        for control in module.controls:
            query[control.feature] = render_control(control, pool, restricted)
            # Les commandes categorielles restreignent le vivier des suivantes :
            # un carburant donne n'a pas ete essaye dans toutes les conditions.
            if not pd.api.types.is_numeric_dtype(data.frame[control.feature]):
                pool = pool[pool[control.feature] == query[control.feature]]
                if pool.empty:
                    pool = data.frame

        derived = [c for c in module.controls if c.derived]
        if derived:
            st.caption(
                "Grandeurs calculees : "
                + ", ".join(c.label for c in derived)
                + ". Elles decoulent du melange, elles ne se reglent pas "
                "independamment — toute combinaison n'est pas realisable."
            )

        if choice == "suppression":
            total = query["o2_frac"] + query["co2_frac"] + query["he_frac"]
            if total > 1.0:
                st.error(
                    f"Les fractions molaires somment a {total:.2f}. "
                    "Un melange ne peut pas depasser 1."
                )
                st.stop()
            st.caption(f"azote de complement : {1 - total:.2f}")

    features = to_features(module, query)
    chance = probability(data, features)
    proximity = data.index.assess(features)
    title, meaning = OUTCOME_LABELS[choice]

    with results:
        left, right = st.columns([1, 1])
        with left:
            st.markdown(f"**{title}**")
            opacity = "1" if proximity.trustworthy else ".35"
            st.markdown(
                f"<div style='font-size:52px;font-weight:700;line-height:1;"
                f"opacity:{opacity}'>{chance:.0%}</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                meaning
                if proximity.trustworthy
                else "en retrait : la condition sort du domaine mesure"
            )
        with right:
            st.markdown("**Distance aux donnees**")
            badge(proximity.verdict)
            st.caption(proximity.explain())

        for name, (low, high) in proximity.out_of_range.items():
            st.warning(f"`{name}` : les essais ne couvrent que {low:g} a {high:g}.")

        diameter = predicted_diameter(data, features)
        rate = predicted_rate(data, features)
        if diameter is not None and rate is not None:
            st.markdown("---")
            d0 = float(query["d0_mm"])
            duration = burn_duration(d0, diameter, rate)

            numbers = st.columns(3)
            with numbers[0]:
                st.markdown("**Diametre d'extinction**")
                st.markdown(
                    f"<div style='font-size:30px;font-weight:700;line-height:1.1'>"
                    f"{diameter:.2f} mm</div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"± {data.regressor_error:.2f} · {diameter / d0:.0%} de d0")
            with numbers[1]:
                st.markdown("**Vitesse de combustion**")
                st.markdown(
                    f"<div style='font-size:30px;font-weight:700;line-height:1.1'>"
                    f"{rate:.3f}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"mm²/s · ± {data.rate_error:.3f} · constante K")
            with numbers[2]:
                st.markdown("**Duree jusqu'a l'extinction**")
                st.markdown(
                    f"<div style='font-size:30px;font-weight:700;line-height:1.1'>"
                    f"{duration:.1f} s</div>",
                    unsafe_allow_html=True,
                )
                st.caption("deduite de la loi en d², non mesuree ici")

            components.html(droplet_page(d0, diameter, rate), height=430)
            st.caption(
                "La gouttelette retrecit selon la **loi en d²** — d²(t) = d₀² − K·t — "
                "le resultat fondateur de la combustion de gouttelettes. Ce n'est pas "
                "une interpolation entre deux mesures : K est releve essai par essai, "
                "et la loi reproduit les durees mesurees avec une correlation de "
                "**0.995** sur les 158 extinctions completes. Les grilles bleue et "
                "orange marquent le depart et l'extinction."
            )
            if chance < 0.5:
                st.caption(
                    ":orange[Le modele ne donne que "
                    f"{chance:.0%} de chances d'extinction : cette trajectoire "
                    "decrit un cas qu'il juge peu probable.]"
                )

        st.markdown("---")
        st.markdown("**Essais NASA les plus proches**")
        st.caption(
            "De vraies combustions en microgravite. Elles valent mieux que la "
            "prediction quand elles sont proches."
        )
        shown = ["distance"] + module.features + [module.label]
        st.dataframe(
            proximity.neighbours[[c for c in shown if c in proximity.neighbours]],
            hide_index=True,
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
with tab_coverage:
    st.subheader("Le domaine teste n'est pas une boite")
    st.caption(
        "Chaque case est un croisement de deux conditions. Les cases sombres "
        "n'ont jamais ete essayees : ce n'est pas « peu de donnees », c'est aucune."
    )

    numeric_controls = {
        c.label: c.feature
        for c in module.controls
        if pd.api.types.is_numeric_dtype(data.frame[c.feature])
        and data.frame[c.feature].nunique() <= 40
    }
    if len(numeric_controls) < 2:
        st.info("Ce module n'a pas deux variables discretes a croiser.")
    else:
        names = list(numeric_controls)
        pick = st.columns([1, 1, 2])
        with pick[0]:
            y_name = st.selectbox("Axe vertical", names, index=0, key=f"y_{choice}")
        with pick[1]:
            x_name = st.selectbox(
                "Axe horizontal", names, index=min(1, len(names) - 1), key=f"x_{choice}"
            )

        y_key, x_key = numeric_controls[y_name], numeric_controls[x_name]
        if x_key == y_key:
            st.info("Choisir deux variables differentes.")
        else:
            stats = coverage_stats(data.frame, x_key, y_key)
            st.plotly_chart(
                coverage_figure(
                    data.frame,
                    x_key,
                    y_key,
                    current=(float(query[x_key]), float(query[y_key])),
                ),
                use_container_width=True,
            )
            st.caption(
                f"Le carre orange marque la position choisie dans l'onglet "
                f"Prediction. {stats['filled']} cases remplies sur {stats['cells']}."
            )

    st.markdown("---")
    st.markdown("**Se placer sur un essai reel**")
    nearest = nearest_real_test(data.frame, features, module.features)
    st.dataframe(
        nearest[[c for c in module.features + [module.label] if c in nearest.index]]
        .to_frame("valeur")
        .T,
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "L'essai le plus proche de la position courante. Ses conditions sont, "
        "par construction, dans le domaine teste."
    )

# ---------------------------------------------------------------------------
# Le catalogue : la moitie du livrable, puisque le defi porte sur la
# fragmentation des donnees et non sur un seul jeu.
# ---------------------------------------------------------------------------
with tab_catalogue:
    stats = summary()
    st.subheader("Les investigations NASA, rassemblees")
    st.caption(
        "Le defi porte sur la fragmentation : ces resultats sont publics mais "
        "eparpilles dans des dizaines d'investigations sans format commun. "
        "Savoir lesquelles ne menent nulle part est un resultat — c'est du temps "
        "que la personne suivante n'aura pas a perdre."
    )

    counters = st.columns(5)
    for column, (value, label) in zip(
        counters,
        [
            (stats["total"], "investigations"),
            (stats["exploitable"], "exploitees"),
            (stats["partiel"], "partiellement"),
            (stats["impasse"], "sans resultat par essai"),
            (f"{stats['rows']:,}".replace(",", " "), "lignes exploitables"),
        ],
    ):
        with column:
            st.markdown(
                f"<div style='font-size:30px;font-weight:700;line-height:1.1'>"
                f"{value}</div>",
                unsafe_allow_html=True,
            )
            st.caption(label)

    st.markdown("---")
    wanted = st.multiselect(
        "Filtrer",
        list(VERDICT_LABELS.values()),
        default=list(VERDICT_LABELS.values()),
        label_visibility="collapsed",
    )
    inventory = table()
    st.dataframe(
        inventory[inventory["verdict"].isin(wanted)],
        hide_index=True,
        use_container_width=True,
        column_config={
            "lignes exploitables": st.column_config.NumberColumn(format="%d"),
            "notre evaluation": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        "Les colonnes **plateforme**, **periode** et **titre NASA** sont scannees "
        "depuis les info.md, reproduits verbatim. La colonne **notre evaluation** "
        "est notre travail, pas celui de NASA."
    )

    st.markdown("---")
    st.markdown("**Lire une investigation**")
    items = {i.psi: i for i in scan()}
    picked = st.selectbox(
        "Investigation",
        list(items),
        format_func=lambda k: f"{k} — {VERDICT_LABELS[items[k].verdict]}",
        label_visibility="collapsed",
    )
    item = items[picked]
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("**Ce que NASA dit**")
        for label, key in [
            ("titre", "title"), ("plateforme", "platform"),
            ("debut", "start"), ("fin", "end"),
            ("financeur", "sponsor"), ("centre", "centre"),
        ]:
            if item.nasa.get(key):
                st.caption(f"{label} : {item.nasa[key]}")
        if item.objectives:
            st.markdown(f"> {item.objectives}")
        elif picked == "PSI-159":
            st.warning(
                "L'info.md de PSI-159 est un squelette vide, seul du catalogue. "
                "Il attend que le texte NASA y soit colle verbatim."
            )
    with right:
        st.markdown("**Ce qu'on a trouve**")
        st.caption(item.assessment)
        st.caption(
            f"fichiers : {item.files['csv']} csv · {item.files['reports']} PDF"
            + (f" · {item.files['fields']} grilles" if item.files["fields"] else "")
        )
        if item.rows:
            st.caption(f"{item.rows} lignes exploitables produites")

    with st.expander(f"Texte NASA integral — {picked}"):
        st.markdown(read_info_text(picked) or "_aucun info.md_")

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
with tab_about:
    st.subheader("Pourquoi la probabilite n'est jamais affichee seule")
    st.markdown(
        """
Sur PSI-69, une grille de curseurs a dix crans sur l'oxygene, dix sur le CO2,
huit sur l'helium et huit sur la pression ouvre **6 400 combinaisons**. Quarante
d'entre elles ont ete visitees par au moins un essai reel, soit **0,6 %**.

Le plan d'experience fait varier **un facteur a la fois** : sur le croisement
oxygene x CO2, 31 cases remplies sur 196. D'ou un piege contre-intuitif —
choisir deux valeurs testees separement ne garantit pas que leur combinaison
l'ait ete. Exemple trouve par le voyant lui-meme : FLEX-1 n'a **jamais**
combine CO2 et helium, zero essai sur 213.

Le seuil qui separe « dans le domaine » de « extrapolation » n'est pas choisi a
la main. On mesure d'abord, dans le jeu de donnees, la distance de chaque essai
a son plus proche voisin : cette distribution dit ce qu'est un voisinage normal
pour ces donnees-la, et le seuil se recalibre pour chaque module.
        """
    )
    st.markdown("---")
    st.subheader("Le modele n'est pas le meme d'un module a l'autre")
    st.markdown(
        """
Le plan initial proscrivait les ensembles « a ce nombre de lignes ». Mesure jeu
par jeu, cette regle tient pour l'un et pas pour l'autre :

| module | essais | classe rare | regression | arbre / foret |
|---|---|---|---|---|
| Suppression (PSI-69) | 206 | 27 | **0.753** | arbre illimite 0.678 |
| Auto-entretien (PSI-159) | 272 | 89 | 0.722 | **foret prof. 6 — 0.840** |

Sur PSI-159 la foret gagne douze points, et la raison est physique : les flammes
normales et inverses ont des coefficients de signe **oppose** sur la pression et
le debit. Un modele lineaire qui met les deux configurations en commun moyenne
deux effets contraires ; un arbre separe d'abord sur la configuration.

La foret predit mieux, la regression explique mieux : les deux sont gardees. Le
controle des coefficients contre la physique attendue est ce qui a revele
l'inversion de signe — aucun score ne l'aurait montre.
        """
    )
    st.markdown("---")
    st.caption(
        "Les features n'utilisent que ce qui est connu AVANT l'allumage. Le "
        "diametre d'extinction, la duree de combustion et le taux de combustion "
        "sont mesures pendant ou apres : les donner en entree reviendrait a "
        "predire l'extinction a partir de la preuve qu'elle a eu lieu."
    )
