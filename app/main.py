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
    cross_validated_error,
    predicted_diameter,
    predicted_rate,
    probability,
    to_features,
    value,
)
from flame.loaders.psi99 import load as psi99_load  # noqa: E402
from flame.loaders.psi115 import load as psi115_load  # noqa: E402
from flame.retrieval.search import ReportSearch  # noqa: E402
from flame.dashboard.registry import (  # noqa: E402
    MODULES,
    coverage_summary,
    ready_modules,
)
from flame.models.droplet_burn import burn_duration  # noqa: E402
from flame.viz.droplet_anim import page as droplet_page  # noqa: E402
from flame.viz.cool_flame_live import page as cool_flame_page  # noqa: E402
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
    "cool_flames": ("Probabilite de flamme froide",
                    "une combustion invisible persiste apres l'extinction visible"),
}

SUGGESTIONS = {
    "suppression": [
        "why does carbon dioxide extinguish a droplet flame",
        "what is the d-squared law for droplet burning",
        "radiative extinction of large droplets",
    ],
    "sustainment": [
        "difference between normal and inverse diffusion flames",
        "what is the adiabatic flame temperature",
        "spherical burner flame extinction in microgravity",
    ],
    "detection": [
        "smoke detector response to different materials",
        "particle size distribution of smoke in microgravity",
        "thermal precipitator sampling of smoke particles",
    ],
    "soot": [
        "what is the smoke point of a laminar diffusion flame",
        "soot formation in coflow jet flames",
        "effect of nozzle diameter on flame length",
    ],
    "cool_flames": [
        "what is a cool flame and when does it appear",
        "cool flame extinction diameter versus pressure",
        "low temperature chemistry of dodecane droplets",
    ],
    "materials": [
        "PMMA flame spread in microgravity",
        "SIBAL cotton fiberglass fabric burning",
        "BASS experiment concurrent and opposed flow",
    ],
    "ground": [
        "counterflow burner extinction strain rate",
        "ozone sensitized cool flames",
        "difference between one gravity and microgravity flames",
    ],
}


@st.cache_resource
def bundle_for(key: str):
    return build(MODULES[key])


@st.cache_resource
def threejs_page():
    """La page 3D du panache, si le maillage a ete extrait."""
    from pathlib import Path

    generated = Path("data/figures/psi115_threejs.html")
    if generated.exists():
        return generated.read_text(encoding="utf-8")
    return None


@st.cache_resource
def report_search():
    """L'index TF-IDF, construit une fois pour toutes les sessions."""
    try:
        return ReportSearch()
    except FileNotFoundError:
        return None


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
    "comparables. 24 investigations, 10 exploitables, 7 modules branches."
)

available = ready_modules()
DESCRIPTIVE = {
    key for key, item in available.items() if item.outcome_kind == "descriptive"
}
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

first = "Le catalogue des essais" if choice in DESCRIPTIVE else "Prediction"
tabs = [first, "Ou sont les essais", "Microgravite vs Terre",
        "Le catalogue", "Les rapports", "Modules", "Methode"]
if choice in {"suppression", "cool_flames"}:
    tabs.insert(0, "La flamme, en direct")
rendered = st.tabs(tabs)
if choice in {"suppression", "cool_flames"}:
    (tab_live, tab_predict, tab_coverage, tab_gravity, tab_catalogue,
     tab_reports, tab_modules, tab_about) = rendered
else:
    tab_live = None
    (tab_predict, tab_coverage, tab_gravity, tab_catalogue, tab_reports,
     tab_modules, tab_about) = rendered

# ---------------------------------------------------------------------------
# Onglet interactif : tout se calcule dans le navigateur, donc sans coupure.
# ---------------------------------------------------------------------------
if tab_live is not None and choice == "cool_flames":
    with tab_live:
        st.caption(
            "Trois diametres MESURES, donc trois coquilles emboitees exactes : "
            "la gouttelette au depart, l'extinction de la flamme chaude, celle "
            "de la flamme froide. Entre les deux dernieres il ne se passe rien "
            "de visible a l'oeil, et pourtant la goutte continue de bruler."
        )
        components.html(cool_flame_page(), height=560, scrolling=False)
        st.caption(
            "**L'animation n'est proposee que sur le dodecane pur**, seul "
            "carburant ou la loi en d² a ete verifiee (correlation +0.889). Sur "
            "les melanges dodecane/iso-dodecane elle tombe a +0.03 et +0.50 : "
            "les constituants s'evaporent a des rythmes differents, et sept des "
            "huit gouttelettes dont le diametre d'extinction DEPASSE le diametre "
            "initial sont des melanges — elles ont gonfle avant de bruler. Les "
            "coquilles restent exactes ; seule la trajectoire entre elles serait "
            "inventee."
        )
elif tab_live is not None:
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
if choice in DESCRIPTIVE:
    with tab_predict:
        st.subheader("129 combustions reelles a bord de l'ISS")
        st.warning(
            "**Ce module ne predit rien, et c'est un constat mesure, pas un "
            "renoncement.** Deux pistes d'etiquette ont ete examinees. "
            "L'issue en texte libre n'est ecrite que dans 20 essais sur 129 ; "
            "les 109 autres portent une rampe de debit sans conclusion, et "
            "deduire une extinction d'une liste de nombres serait une "
            "invention. L'oxygene consomme donne quant a lui des valeurs "
            "physiquement impossibles : l'O2 final monte jusqu'a 40.4 %, la "
            "chambre ayant ete repurgee entre les deux releves sur une partie "
            "des essais."
        )
        st.caption(
            "Ce qui reste est precieux : un catalogue de materiaux reellement "
            "embarques, brules en microgravite, avec leur geometrie demelee de "
            "51 orthographes differentes."
        )

        catalogue = data.frame
        picks = st.columns(3)
        with picks[0]:
            families = st.multiselect(
                "Famille", sorted(catalogue["material_family"].dropna().unique()),
                default=sorted(catalogue["material_family"].dropna().unique()),
            )
        with picks[1]:
            forms = st.multiselect(
                "Forme", sorted(catalogue["material_form"].dropna().unique()),
                default=sorted(catalogue["material_form"].dropna().unique()),
            )
        with picks[2]:
            only_stated = st.toggle(
                "Seulement les issues declarees", value=False,
                help="Les 20 essais ou NASA a ecrit ce qui s'est passe.",
            )

        filtered = catalogue[
            catalogue["material_family"].isin(families)
            & catalogue["material_form"].isin(forms)
        ]
        if only_stated:
            filtered = filtered[filtered["outcome"].notna()]

        counters = st.columns(4)
        for column, (number, label) in zip(counters, [
            (len(filtered), "essais"),
            (int(filtered["outcome"].notna().sum()), "issues declarees"),
            (filtered["material_family"].nunique(), "familles"),
            (int(filtered["principal_investigator"].nunique()), "investigateurs"),
        ]):
            with column:
                st.markdown(
                    f"<div style='font-size:28px;font-weight:700;line-height:1.1'>"
                    f"{number}</div>", unsafe_allow_html=True)
                st.caption(label)

        st.dataframe(
            filtered[[
                "test_id", "principal_investigator", "material_family",
                "material_form", "thickness_mm", "width_cm", "diameter_mm",
                "exposed_faces", "o2_initial_pct", "outcome", "material_raw",
            ]].rename(columns={
                "test_id": "essai", "principal_investigator": "investigateur",
                "material_family": "famille", "material_form": "forme",
                "thickness_mm": "ep. mm", "width_cm": "larg. cm",
                "diameter_mm": "diam. mm", "exposed_faces": "faces",
                "o2_initial_pct": "O2 %", "outcome": "issue",
                "material_raw": "texte NASA d'origine",
            }),
            hide_index=True, use_container_width=True, height=420,
        )
        st.caption(
            "La colonne « texte NASA d'origine » montre ce qu'il a fallu "
            "demeler : le meme echantillon y est ecrit « 2 cm 100 micron thick "
            "PMMA », « 100 micron PMMA film 2 cm wide » et « 100 micron film "
            "2 cm wide »."
        )

        st.markdown("---")
        st.markdown("**Qualite des mesures de gaz**")
        flags = st.columns(3)
        for column, (count, label, why) in zip(flags, [
            (int(catalogue["o2_gained"].sum()), "O2 final > O2 initial",
             "physiquement impossible en chambre close"),
            (int(catalogue["co_negative"].sum()), "CO negatif",
             "decalage de capteur"),
            (int(catalogue["flow_ramp_ambiguous"].sum()), "rampe ambigue",
             "valeurs d'un autre instrument melees"),
        ]):
            with column:
                st.markdown(
                    f"<div style='font-size:26px;font-weight:700;line-height:1.1'>"
                    f"{count}</div>", unsafe_allow_html=True)
                st.caption(f"{label} — {why}")
else:
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
    proximity = data.index.assess(features)
    regression = module.outcome_kind == "regression"

    if regression:
        chance = 0.0
        prediction = value(data, features)
        spread = cross_validated_error(data)
        title = module.target_label
        meaning = module.target_meaning
        shown_value = f"{prediction:.2f} {module.target_unit}"
    else:
        chance = probability(data, features)
        title, meaning = OUTCOME_LABELS[choice]
        shown_value = f"{chance:.0%}"

    with results:
        left, right = st.columns([1, 1])
        with left:
            st.markdown(f"**{title}**")
            opacity = "1" if proximity.trustworthy else ".35"
            st.markdown(
                f"<div style='font-size:46px;font-weight:700;line-height:1;"
                f"opacity:{opacity}'>{shown_value}</div>",
                unsafe_allow_html=True,
            )
            if regression:
                st.caption(
                    f"± {spread:.2f} {module.target_unit}, l'erreur moyenne du "
                    "modele en validation croisee"
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
        shown = ["distance"] + module.features + [module.label or module.target]
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
# Le contraste gravite / microgravite : la question de fond du domaine.
# ---------------------------------------------------------------------------
with tab_gravity:
    st.subheader("Le feu dans l'espace n'est pas le feu sur Terre")
    st.caption(
        "Deux sources du catalogue repondent directement a cette question. "
        "L'une est une mesure sur neuf echantillons, l'autre une simulation."
    )

    st.markdown("### Le meme materiau, en orbite et au sol")
    st.caption(
        "PSI-99 / Saffire-II. Neuf echantillons seulement, bien trop peu pour "
        "entrainer quoi que ce soit — mais c'est la SEULE source du catalogue "
        "qui porte les deux comportements sur la meme ligne."
    )
    saffire = psi99_load()
    view = saffire[[
        "sample_id", "material", "thickness_mm", "flow_direction",
        "ug_burn_length_raw", "ug_spread_rate_raw",
        "g1_burn_length_raw", "g1_spread_rate_raw",
    ]].rename(columns={
        "sample_id": "ech.", "material": "materiau", "thickness_mm": "ep. mm",
        "flow_direction": "ecoulement",
        "ug_burn_length_raw": "microgravite : longueur",
        "ug_spread_rate_raw": "microgravite : vitesse",
        "g1_burn_length_raw": "1 g : longueur",
        "g1_spread_rate_raw": "1 g : vitesse",
    })
    st.dataframe(view, hide_index=True, use_container_width=True)

    burned = int(saffire["burned_in_microgravity"].sum())
    left, right = st.columns(2)
    with left:
        st.markdown("**Le silicone**")
        st.markdown(
            "<div style='font-size:34px;font-weight:700;line-height:1.1'>"
            "0 / 4</div>", unsafe_allow_html=True)
        st.caption(
            "echantillons ayant brule en microgravite. A 1 g, deux d'entre eux "
            "brulent COMPLETEMENT. Un materiau peut donc etre dangereux au sol "
            "et inerte en orbite."
        )
    with right:
        st.markdown("**Le tissu SIBAL**")
        st.markdown(
            "<div style='font-size:34px;font-weight:700;line-height:1.1'>"
            "2,1 - 2,6 mm/s</div>", unsafe_allow_html=True)
        st.caption(
            "vitesse de propagation en microgravite, constante. A 1 g la colonne "
            "porte « Acceleratory » : il n'y a pas de vitesse stable a donner. "
            "La flottabilite emballe la flamme sur Terre ; en apesanteur elle "
            "avance regulierement."
        )
    st.caption(
        f"Au total, {burned} des {len(saffire)} echantillons ont brule en "
        "microgravite. Ces neuf lignes ne se modelisent pas : elles se lisent."
    )

    st.markdown("---")
    st.markdown("### Le panache de fumee, avec et sans gravite")
    st.caption(
        "PSI-115. Simulation numerique, pas une mesure. Le volume de gauche est "
        "obtenu en faisant tourner la tranche calculee autour de son axe : le "
        "cas en microgravite est axisymetrique (asymetrie mesuree 0.0002), donc "
        "cette revolution reconstruit le volume que la simulation representait "
        "deja. Le cas terrestre ne l'est pas (0.1133) : la gravite designe une "
        "direction, le panache devie, et le revolutionner produirait une forme "
        "sans realite physique."
    )
    fields = psi115_load()
    compare = fields[
        fields["variable"].isin(["smoke", "vort", "u", "numden"])
        & fields["thermophoresis"]
    ].pivot_table(index="variable_label", columns="gravity", values="max").round(2)
    compare.columns = [f"maximum a {c}" for c in compare.columns]
    st.dataframe(compare, use_container_width=True)
    st.caption(
        "La vorticite chute de 56.8 a 35.2 et la vitesse axiale de 3.4 a 2.4 : "
        "c'est la flottabilite qui disparait. Mais la fraction de fumee MONTE, "
        "de 9.4 a 12.9. Sans courant ascendant pour l'emporter, la fumee ne "
        "part pas : elle stagne pres de la source. Unites normalisees, non "
        "metriques."
    )

    with st.expander("Voir le panache en 3D"):
        page = threejs_page()
        if page:
            components.html(page, height=620, scrolling=False)
        else:
            st.info(
                "La page 3D n'est pas generee. La produire avec "
                "`py -m flame.viz.threejs` (necessite mesh_axes.csv)."
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
# Les rapports : repondre avec ses sources, ou se taire.
# ---------------------------------------------------------------------------
with tab_reports:
    engine = report_search()
    if engine is None:
        st.warning(
            "Le corpus n'est pas construit. Le produire avec "
            "`py scripts/extract_pdf_text.py`."
        )
    else:
        st.subheader("Chercher dans les rapports NASA")
        st.caption(
            f"{len(engine):,} passages extraits de "
            f"{engine.corpus['document'].nunique()} rapports. Chaque resultat "
            "porte son document et sa page : **la recherche ne repond que ce "
            "qu'elle peut citer**, et se tait quand elle ne trouve rien."
        )

        chips = st.columns(len(SUGGESTIONS[choice]))
        for column, suggestion in zip(chips, SUGGESTIONS[choice]):
            with column:
                if st.button(suggestion[:38] + "…", key=f"sug_{suggestion}",
                             use_container_width=True):
                    st.session_state["question"] = suggestion

        question = st.text_input(
            "Question",
            key="question",
            placeholder="en anglais — les rapports le sont",
            label_visibility="collapsed",
        )

        picker = st.columns([2, 2])
        with picker[0]:
            narrow = st.selectbox(
                "Limiter a une investigation",
                ["toutes"] + sorted(engine.corpus["investigation"].unique()),
            )
        with picker[1]:
            mode = st.radio(
                "Mecanisme",
                ["Hybride (recommande)", "TF-IDF seul"],
                horizontal=True,
                help=(
                    "Hybride : TF-IDF decide s'il faut repondre, une reduction "
                    "de dimension classe les resultats. Sur huit questions "
                    "paraphrasees, l'hybride trouve la bonne investigation "
                    "6 fois contre 3 pour TF-IDF seul."
                ),
            )
        semantic = mode.startswith("Hybride")

        if question:
            hits = engine.search(
                question, limit=6,
                investigation=None if narrow == "toutes" else narrow,
                semantic=semantic,
            )
            if not hits:
                st.info(
                    "Aucun passage assez proche. La question sort du corpus, ou "
                    "son vocabulaire n'y figure pas — la recherche compare des "
                    "mots, pas des idees."
                )
            for hit in hits:
                with st.container(border=True):
                    head, score = st.columns([4, 1])
                    with head:
                        st.markdown(f"**{hit.citation}**")
                    with score:
                        st.markdown(
                            f"<div style='text-align:right;color:#898781'>"
                            f"{hit.score:.3f}</div>",
                            unsafe_allow_html=True,
                        )
                    if hit.terms:
                        st.caption("termes partages : " + ", ".join(hit.terms))
                    else:
                        st.caption(
                            ":orange[aucun terme commun] — ce resultat vient du "
                            "rapprochement semantique, pas d'un mot partage"
                        )
                    st.markdown(f"> {hit.text}")

        st.markdown("---")
        with st.expander("Ce que le corpus contient — et ne contient pas"):
            st.dataframe(engine.coverage(), hide_index=True,
                         use_container_width=True)
            st.caption(
                "Onze investigations sur 24 ont des rapports ; les autres n'en "
                "ont pas, et aucune recherche ne fera apparaitre ce qui n'a pas "
                "ete publie. L'extraction perd par ailleurs la mise en page : "
                "pour un chiffre, les tables de data/processed/ sont la source, "
                "pas ce corpus."
            )
            st.markdown("**Deux mecanismes montes en serie**")
            st.caption(
                "TF-IDF compare des mots : il est aveugle aux reformulations et "
                "ne trouve la bonne investigation que 3 fois sur 8 questions "
                "paraphrasees. Une reduction de dimension y arrive 6 fois, mais "
                "elle PERD le droit de se taire : seule, elle donnait 0.84 a "
                "« quelle est la capitale de l'Australie », plus qu'a la plupart "
                "des vraies questions."
            )
            st.caption(
                "D'ou le montage : TF-IDF tient la porte, la reduction fait le "
                "classement. Une question doit partager au moins deux termes "
                "substantiels avec le corpus pour qu'on y reponde. Ce seuil "
                "garde 8 questions du domaine sur 8 et rejette 6 hors-sujet "
                "sur 7."
            )
            st.caption(
                "**La limite reste reelle.** « stock market prices today » passe "
                "encore, ses deux mots existant dans le corpus. Aucune "
                "statistique lexicale ne distingue parfaitement le hors-sujet : "
                "c'est pourquoi les termes partages sont affiches a cote de "
                "chaque resultat. La transparence complete le filtre, elle ne "
                "le remplace pas."
            )
            st.caption(
                "Ce n'est dans aucun cas un modele de langage. Un modele "
                "generatif produirait des reponses plausibles et invérifiables ; "
                "dans un outil de securite incendie, c'est disqualifiant. "
                "`py scripts/benchmark_search.py` rejoue la comparaison."
            )

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
