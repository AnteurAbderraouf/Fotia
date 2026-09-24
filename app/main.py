"""Fotia — NASA microgravity combustion data, made comparable.

    py -m streamlit run app/main.py

L'INTERFACE EST EN ANGLAIS, LE CODE EN FRANCAIS. Le tableau de bord s'adresse
a un jury Space Apps international et a quiconque trouve le depot ; les
commentaires et la documentation restent en francais, pour l'auteur.

TROIS NIVEAUX, DU SIMPLE AU VERIFIABLE.

    01  What we found    quatre constats chiffres, sans jargon ni numero PSI.
                         Ce qu'un jury voit en cinq minutes.
    02  Ask a question   les sept modules regroupes en six questions humaines.
                         On entre par ce qu'on veut savoir, pas par le regime.
    03  Check the work   le catalogue, les cartes de couverture, la recherche
                         dans les rapports, le detail des modeles.

Le tableau de bord etait auparavant organise comme on l'avait construit, par
investigation NASA. Un visiteur ne pense pas « je veux PSI-159 » : il pense
« est-ce que ce materiau est dangereux ». Le numero PSI ne disparait pas, il
passe en source.

LES AVERTISSEMENTS SONT DEVENUS UN FEU TRICOLORE. Ils occupaient des
paragraphes entiers sous chaque prediction, ce qui, pour un lecteur non
specialiste, se lit comme une suite d'excuses. Ils tiennent maintenant dans un
badge au vocabulaire FIXE, Measured here / Between tests / Never tested, et
le detail reste accessible d'un clic. La rigueur ne disparait pas, elle change
de format.
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
    tested_values,
)
from flame.dashboard.findings import headline  # noqa: E402
from flame.dashboard.predict import (  # noqa: E402
    build,
    cross_validated_error,
    predicted_diameter,
    predicted_rate,
    probability,
    to_features,
    value,
)
from flame.dashboard.questions import CONFIDENCE, QUESTIONS  # noqa: E402
from flame.dashboard.registry import MODULES, coverage_summary, ready_modules  # noqa: E402
from flame.loaders.psi99 import load as load_saffire  # noqa: E402
from flame.loaders.psi115 import load as load_fields  # noqa: E402
from flame.models.droplet_burn import burn_duration  # noqa: E402
from flame.retrieval.search import ReportSearch  # noqa: E402
from flame.viz.cool_flame_live import page as cool_flame_page  # noqa: E402
from flame.viz.droplet_live import page as droplet_page  # noqa: E402
from flame.viz.jet_flame_live import page as jet_flame_page  # noqa: E402
from flame.viz.saffire_live import page as saffire_page  # noqa: E402

st.set_page_config(page_title="Fotia", layout="wide", page_icon="🔥")

TONES = {
    "good": "#35b37e",
    "warning": "#e0a327",
    "critical": "#d9534f",
    "neutral": "#7d8795",
}

OUTCOME_LABELS = {
    "suppression": ("Chance of extinction", "the flame goes out"),
    "sustainment": ("Chance of self-extinction", "the flame dies on its own"),
    "cool_flames": ("Chance of a cool flame", "invisible burning continues"),
}

SUGGESTIONS = {
    "suppression": [
        "why does carbon dioxide extinguish a droplet flame",
        "what is the d-squared law for droplet burning",
    ],
    "cool_flames": [
        "what is a cool flame and when does it appear",
        "cool flame extinction diameter versus pressure",
    ],
    "sustainment": [
        "difference between normal and inverse diffusion flames",
        "spherical burner flame extinction in microgravity",
    ],
    "detection": [
        "smoke detector response to different materials",
        "soot formation in coflow jet flames",
    ],
    "soot": ["what is the smoke point of a laminar diffusion flame"],
    "materials": ["PMMA flame spread in microgravity"],
    "ground": ["counterflow burner extinction strain rate"],
}

# La vue de la goutte est la seule a recevoir le module entraine : elle
# embarque les coefficients dans la page pour recalculer dans le navigateur,
# sans quoi chaque mouvement de curseur relancerait Streamlit et couperait
# l'animation. Les trois autres n'affichent que des mesures.
LIVE_VIEWS = {
    "suppression": (lambda data: droplet_page(data), "The droplet, live"),
    "cool_flames": (lambda data: cool_flame_page(), "The three shells"),
    "soot": (lambda data: jet_flame_page(), "The jet flame"),
}


# ---------------------------------------------------------------------------
# Chargements, mis en cache une fois pour toutes les sessions
# ---------------------------------------------------------------------------
@st.cache_resource
def bundle_for(key: str):
    """Le module entraine. RESERVE AUX MODULES PREDICTIFS.

    `build` exige une etiquette ou une cible ; le module Materiaux n'en a
    aucune, par constat mesure et non par oubli. Il passe donc par
    `frame_for`, qui charge la table sans rien entrainer dessus.
    """
    return build(MODULES[key])


@st.cache_data
def frame_for(key: str) -> pd.DataFrame:
    """La table brute d'un module, sans modele."""
    return MODULES[key].loader()


@st.cache_resource
def report_search():
    try:
        return ReportSearch()
    except FileNotFoundError:
        return None


@st.cache_resource
def findings():
    return headline(bundle_for("suppression"))


@st.cache_resource
def threejs_page():
    generated = Path("data/figures/psi115_threejs.html")
    return generated.read_text(encoding="utf-8") if generated.exists() else None


# ---------------------------------------------------------------------------
# Elements d'interface partages
# ---------------------------------------------------------------------------
def confidence_badge(verdict: str) -> None:
    """Le feu tricolore, au vocabulaire fixe partout dans l'interface."""
    label, tone, _ = CONFIDENCE[verdict]
    st.markdown(
        f"<span style='background:{TONES[tone]};color:#0d0d0d;padding:5px 11px;"
        f"border-radius:4px;font-weight:700;font-size:12px;letter-spacing:.04em;"
        f"text-transform:uppercase'>{label}</span>",
        unsafe_allow_html=True,
    )


def big_number(text: str, tone: str = "neutral", dim: bool = False) -> None:
    colour = TONES.get(tone, "inherit") if tone != "neutral" else "inherit"
    st.markdown(
        f"<div style='font-size:44px;font-weight:700;line-height:1.05;"
        f"color:{colour};opacity:{'.35' if dim else '1'}'>{text}</div>",
        unsafe_allow_html=True,
    )


def go(level: str, question: str | None = None) -> None:
    st.session_state["level"] = level
    if question:
        st.session_state["question"] = question


st.session_state.setdefault("level", "01")
st.session_state.setdefault("question", "suppression")

# ---------------------------------------------------------------------------
st.markdown(
    "<div style='display:flex;align-items:baseline;gap:13px;flex-wrap:wrap'>"
    "<span style='font-size:25px;font-weight:700'>Fotia</span>"
    "<span style='color:#8a93a0'>NASA microgravity combustion data, "
    "made comparable</span></div>",
    unsafe_allow_html=True,
)

LEVELS = {
    "01": "What we found",
    "02": "Ask a question",
    "03": "Check the work",
}
level = st.radio(
    "Level",
    list(LEVELS),
    format_func=lambda key: f"{key}  {LEVELS[key]}",
    horizontal=True,
    label_visibility="collapsed",
    key="level",
)
st.divider()


# ===========================================================================
# NIVEAU 01 — ce qu'un jury voit en cinq minutes
# ===========================================================================
if level == "01":
    st.markdown("### Fire behaves differently in space. Here is what the data shows.")
    # La legende disait « chaque chiffre vient d'un essai reel ». C'etait faux :
    # le troisieme constat vient d'une simulation. La carte le disait deja dans
    # sa source, mais la legende la contredisait juste au-dessus.
    st.caption(
        "Four findings, drawn from 24 NASA investigations rebuilt into "
        "comparable tables. Two come from real burns in orbit, one from a "
        "numerical simulation that says so on its card, and the last counts "
        "what was never measured at all."
    )

    # La cle du bouton est l'INDICE du constat, pas sa question : deux constats
    # peuvent mener a la meme question, et Streamlit refuse deux widgets de
    # meme cle. C'est exactement le cas ici, ou le materiau invisible et la
    # fumee qui stagne renvoient tous deux a la detection.
    for row_start in (0, 2):
        columns = st.columns(2, gap="medium")
        for position, (column, finding) in enumerate(
            zip(columns, findings()[row_start : row_start + 2]), start=row_start
        ):
            with column:
                with st.container(border=True):
                    st.markdown(
                        f"<div style='font-size:34px;font-weight:700;line-height:1;"
                        f"color:{TONES[finding.tone]};font-variant-numeric:tabular-nums'>"
                        f"{finding.figure}"
                        f"<span style='font-size:16px;color:#8a93a0;font-weight:500'>"
                        f" {finding.unit}</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**{finding.title}**")
                    st.caption(finding.body)
                    st.caption(f":gray[{finding.source}]")
                    st.button(
                        "Open this question  →",
                        key=f"go_{position}_{finding.question}",
                        width="stretch",
                        on_click=go,
                        args=("02", finding.question),
                    )

    st.divider()
    stats = summary()
    counters = st.columns(4)
    for column, (number, label) in zip(
        counters,
        [
            (stats["total"], "investigations catalogued"),
            (stats["exploitable"] + stats["partiel"], "yielded usable data"),
            (f"{stats['rows']:,}".replace(",", " "), "test rows rebuilt"),
            (stats["impasse"], "led nowhere, and we say so"),
        ],
    ):
        with column:
            st.markdown(
                f"<div style='font-size:26px;font-weight:700;line-height:1.1'>"
                f"{number}</div>",
                unsafe_allow_html=True,
            )
            st.caption(label)


# ===========================================================================
# NIVEAU 02 — entrer par la question
# ===========================================================================
elif level == "02":
    ready = ready_modules()
    labels = {key: q.label for key, q in QUESTIONS.items()}
    question_key = st.radio(
        "Question",
        list(QUESTIONS),
        format_func=lambda key: labels[key],
        key="question",
        label_visibility="collapsed",
        horizontal=False,
    )
    question = QUESTIONS[question_key]

    st.markdown(f"### {question.label}")
    sources = ", ".join(
        ready[m].investigation for m in question.modules if m in ready
    )
    st.caption(f"{question.blurb}  ·  :gray[{sources}]")

    available = [m for m in question.modules if m in ready]
    module_key = available[0]
    if len(available) > 1:
        module_key = st.radio(
            "Dataset",
            available,
            format_func=lambda key: f"{ready[key].name} · {ready[key].investigation}",
            horizontal=True,
            label_visibility="collapsed",
        )
    module = ready[module_key]

    # --- module descriptif : il ne predit rien, et il le dit ----------------
    if module.outcome_kind == "descriptive":
        st.warning(
            "**This dataset cannot predict, and that is a measured finding, "
            "not a shortcut.** Only 20 of its 129 burns state an outcome in "
            "writing; the other 109 record an airflow ramp with no conclusion. "
            "Oxygen consumption looked promising but reads as physically "
            "impossible in places, with final O2 reaching 40.4 %. The chamber "
            "was purged between readings on some runs."
        )
        st.caption(
            "What remains is worth having: 129 real ISS burns with each "
            "sample's geometry untangled from 49 different spellings of the "
            "material name."
        )
        catalogue = frame_for(module_key)
        picks = st.columns(2)
        with picks[0]:
            families = st.multiselect(
                "Material family",
                sorted(catalogue["material_family"].dropna().unique()),
                default=sorted(catalogue["material_family"].dropna().unique()),
            )
        with picks[1]:
            only_stated = st.toggle(
                "Only burns with a stated outcome",
                help="The 20 runs where NASA wrote down what happened.",
            )
        filtered = catalogue[catalogue["material_family"].isin(families)]
        if only_stated:
            filtered = filtered[filtered["outcome"].notna()]
        st.dataframe(
            filtered[[
                "test_id", "principal_investigator", "material_family",
                "material_form", "thickness_mm", "diameter_mm",
                "o2_initial_pct", "outcome", "material_raw",
            ]].rename(columns={
                "test_id": "run", "principal_investigator": "investigator",
                "material_family": "family", "material_form": "form",
                "thickness_mm": "thickness mm", "diameter_mm": "diameter mm",
                "o2_initial_pct": "O2 %", "outcome": "outcome",
                "material_raw": "NASA's own wording",
            }),
            hide_index=True, width="stretch", height=380,
        )

        st.divider()
        st.markdown("#### The same material, in orbit and on the ground")
        st.caption(
            "Saffire-II. Nine samples, far too few to train on, but the only "
            "source in the catalogue carrying both behaviours on one row. Each "
            "strip is 29 by 5 cm; the dark portion is the length that actually "
            "burned."
        )
        components.html(saffire_page(), height=470, scrolling=False)

    # --- module predictif ---------------------------------------------------
    else:
        data = bundle_for(module_key)
        controls, results = st.columns([1, 2], gap="large")
        with controls:
            restricted = st.toggle(
                "Tested conditions only",
                value=True,
                key=f"restrict_{module_key}",
                help=(
                    "Limits each control to values NASA actually ran. Careful: "
                    "two separately tested values can still form a combination "
                    "nobody ever tried."
                ),
            )
            query: dict = {}
            pool = data.frame
            for control in module.controls:
                series = pool[control.feature]
                caption = control.label + (f" ({control.unit})" if control.unit else "")
                if not pd.api.types.is_numeric_dtype(series):
                    query[control.feature] = st.selectbox(
                        caption, sorted(series.dropna().unique()),
                        help=control.help or None,
                    )
                    pool = pool[pool[control.feature] == query[control.feature]]
                    if pool.empty:
                        pool = data.frame
                    continue
                values = tested_values(pool, control.feature)
                low, high = float(series.min()), float(series.max())
                if restricted and len(values) <= 20:
                    query[control.feature] = st.select_slider(
                        caption, values, value=values[len(values) // 2],
                        help=control.help or None,
                    )
                elif restricted:
                    query[control.feature] = st.slider(
                        caption, low, high, float(series.median()),
                        help=control.help or None,
                    )
                else:
                    margin = (high - low) * 0.35 or 1.0
                    query[control.feature] = st.slider(
                        caption, low - margin, high + margin,
                        float(series.median()), help=control.help or None,
                    )

            derived = [c for c in module.controls if c.derived]
            if derived:
                st.caption(
                    "Computed quantities: " + ", ".join(c.label for c in derived)
                    + ". They follow from the mixture and cannot be set "
                    "independently, so not every combination is physically real."
                )
            if module_key == "suppression":
                total = query["o2_frac"] + query["co2_frac"] + query["he_frac"]
                if total > 1.0:
                    st.error(
                        f"Mole fractions add up to {total:.2f}. A mixture "
                        "cannot exceed 1."
                    )
                    st.stop()
                st.caption(f"nitrogen makes up the rest: {1 - total:.2f}")

        features = to_features(module, query)
        proximity = data.index.assess(features)
        regression = module.outcome_kind == "regression"

        with results:
            left, right = st.columns([1, 1])
            with left:
                if regression:
                    spread = cross_validated_error(data)
                    st.markdown(f"**{module.target_label}**")
                    big_number(
                        f"{value(data, features):.2f} {module.target_unit}",
                        dim=not proximity.trustworthy,
                    )
                    st.caption(
                        f"± {spread:.2f} {module.target_unit}, the model's mean "
                        "error in cross-validation"
                    )
                    st.caption(module.target_meaning)
                else:
                    chance = probability(data, features)
                    title, meaning = OUTCOME_LABELS[module_key]
                    st.markdown(f"**{title}**")
                    big_number(f"{chance:.0%}", dim=not proximity.trustworthy)
                    st.caption(
                        meaning
                        if proximity.trustworthy
                        else "dimmed: this condition sits outside what was measured"
                    )
            with right:
                st.markdown("**How close is real data?**")
                confidence_badge(proximity.verdict)
                st.caption(CONFIDENCE[proximity.verdict][2])
                with st.expander("Why this reading?"):
                    st.caption(proximity.explain())
                    st.caption(
                        "The threshold is not hand-picked. We first measure how "
                        "far each real test sits from its own nearest neighbour; "
                        "that distribution defines what a normal neighbourhood "
                        "is for this dataset, and it recalibrates itself for "
                        "every module."
                    )
                    for name, (low, high) in proximity.out_of_range.items():
                        st.warning(f"`{name}`: tests only cover {low:g} to {high:g}.")

            if module_key == "suppression":
                diameter = predicted_diameter(data, features)
                rate = predicted_rate(data, features)
                if diameter is not None and rate is not None:
                    d0 = float(query["d0_mm"])
                    extras = st.columns(3)
                    for column, (figure, label, note) in zip(extras, [
                        (f"{diameter:.2f} mm", "Extinction diameter",
                         f"± {data.regressor_error:.2f} · {diameter / d0:.0%} of the start"),
                        (f"{rate:.3f}", "Burning rate K",
                         f"mm²/s · ± {data.rate_error:.3f}"),
                        (f"{burn_duration(d0, diameter, rate):.1f} s", "Time to extinction",
                         "from the d² law, not measured here"),
                    ]):
                        with column:
                            st.markdown(
                                f"<div style='font-size:24px;font-weight:700;"
                                f"line-height:1.1'>{figure}</div>",
                                unsafe_allow_html=True,
                            )
                            st.caption(f"**{label}**")
                            st.caption(note)

            st.markdown("**Closest real NASA tests**")
            st.caption(
                "Actual burns in microgravity. When they are close, they are "
                "worth more than the prediction."
            )
            shown = ["distance"] + module.features + [module.label or module.target]
            st.dataframe(
                proximity.neighbours[
                    [c for c in shown if c in proximity.neighbours]
                ],
                hide_index=True, width="stretch",
            )

        # --- vue 3D du module, si elle existe -------------------------------
        if module_key in LIVE_VIEWS:
            builder, title = LIVE_VIEWS[module_key]
            st.divider()
            st.markdown(f"#### {title}")
            components.html(builder(data), height=545, scrolling=False)

        # --- recherche ciblee ------------------------------------------------
        engine = report_search()
        if engine is not None:
            st.divider()
            st.markdown("#### What the reports say")
            st.caption(
                "Passages from NASA's own reports, each with its document and "
                "page. The search answers only what it can cite."
            )
            chips = st.columns(len(SUGGESTIONS.get(module_key, [])) or 1)
            for column, suggestion in zip(chips, SUGGESTIONS.get(module_key, [])):
                with column:
                    if st.button(suggestion[:44] + "…", key=f"s_{suggestion}",
                                 width="stretch"):
                        st.session_state["report_query"] = suggestion
            typed = st.text_input(
                "Search", key="report_query",
                placeholder="Ask in English; the reports are in English",
                label_visibility="collapsed",
            )
            if typed:
                hits = engine.search(typed, limit=3)
                if not hits:
                    st.info(
                        "Nothing close enough. The question sits outside the "
                        "corpus, or its wording does not appear in it."
                    )
                for hit in hits:
                    with st.container(border=True):
                        st.markdown(f"**{hit.citation}**")
                        if hit.terms:
                            st.caption("shared terms: " + ", ".join(hit.terms))
                        else:
                            st.caption(
                                ":orange[no shared word]: this came from the "
                                "semantic ranking"
                            )
                        st.markdown(f"> {hit.text[:600]}…")


# ===========================================================================
# NIVEAU 03 — tout ce qui permet de verifier
# ===========================================================================
else:
    tabs = st.tabs([
        "The catalogue", "Test coverage", "Gravity contrast",
        "Search the reports", "Model choices",
    ])

    # --- catalogue ---------------------------------------------------------
    with tabs[0]:
        stats = summary()
        st.markdown("#### Every NASA investigation, gathered")
        st.caption(
            "The challenge is about fragmentation: these results are public but "
            "scattered across dozens of investigations with no common format. "
            "Knowing which ones lead nowhere is a result: it is time the next "
            "person will not spend."
        )
        counters = st.columns(4)
        for column, (number, label) in zip(counters, [
            (stats["total"], "investigations"),
            (stats["exploitable"], "fully used"),
            (stats["partiel"], "partly usable"),
            (stats["impasse"], "no per-test outcome"),
        ]):
            with column:
                st.markdown(
                    f"<div style='font-size:26px;font-weight:700;line-height:1.1'>"
                    f"{number}</div>", unsafe_allow_html=True)
                st.caption(label)

        wanted = st.multiselect(
            "Filter", list(VERDICT_LABELS.values()),
            default=list(VERDICT_LABELS.values()), label_visibility="collapsed",
        )
        inventory = table()
        st.dataframe(
            inventory[inventory["status"].isin(wanted)],
            hide_index=True, width="stretch",
        )
        st.caption(
            "Platform, period and title are scanned verbatim from NASA's own "
            "description files. The assessment column is ours, not NASA's."
        )
        items = {i.psi: i for i in scan()}
        picked = st.selectbox("Read one", list(items))
        with st.expander(f"NASA's full text for {picked}"):
            st.markdown(read_info_text(picked) or "_no description file_")

    # --- couverture ---------------------------------------------------------
    with tabs[1]:
        st.markdown("#### The tested domain is not a box")
        st.caption(
            "Each cell is one pair of conditions. Dark cells were never run: "
            "that is not *little* data, it is none. The design varied one "
            "factor at a time; it never swept a grid."
        )
        # Seuls les modules qui ont des commandes ont un plan d'experience a
        # cartographier. Le module Materiaux n'en a pas, et il serait etrange
        # de le proposer pour l'ecarter ensuite.
        ready = {k: m for k, m in ready_modules().items() if m.controls}
        which = st.selectbox(
            "Dataset", list(ready),
            format_func=lambda k: f"{ready[k].name} · {ready[k].investigation}",
        )
        picked_bundle = bundle_for(which)
        numeric = {
            c.label: c.feature
            for c in ready[which].controls
            if pd.api.types.is_numeric_dtype(picked_bundle.frame[c.feature])
            and picked_bundle.frame[c.feature].nunique() <= 40
        }
        if len(numeric) < 2:
            st.info("This dataset has no two discrete variables to cross.")
        else:
            names = list(numeric)
            axes = st.columns(2)
            with axes[0]:
                y_name = st.selectbox("Vertical axis", names, index=0)
            with axes[1]:
                x_name = st.selectbox(
                    "Horizontal axis", names, index=min(1, len(names) - 1)
                )
            if numeric[x_name] == numeric[y_name]:
                st.info("Pick two different variables.")
            else:
                st.plotly_chart(
                    coverage_figure(
                        picked_bundle.frame, numeric[x_name], numeric[y_name]
                    ),
                    width="stretch",
                )
                found = coverage_stats(
                    picked_bundle.frame, numeric[x_name], numeric[y_name]
                )
                st.caption(
                    f"{found['filled']} of {found['cells']} cells were visited "
                    f"({found['share']:.0%})."
                )

    # --- contraste gravite ---------------------------------------------------
    with tabs[2]:
        st.markdown("#### Fire in orbit is not fire on Earth")
        saffire = load_saffire()
        st.caption(
            "Saffire-II carries both regimes on the same row. Nine samples, "
            "which is far too few to model and exactly enough to read."
        )
        st.dataframe(
            saffire[[
                "sample_id", "material", "thickness_mm", "flow_direction",
                "ug_burn_length_raw", "ug_spread_rate_raw",
                "g1_burn_length_raw", "g1_spread_rate_raw",
            ]].rename(columns={
                "sample_id": "sample", "thickness_mm": "thickness mm",
                "flow_direction": "flow",
                "ug_burn_length_raw": "microgravity: length",
                "ug_spread_rate_raw": "microgravity: rate",
                "g1_burn_length_raw": "1 g: length",
                "g1_spread_rate_raw": "1 g: rate",
            }),
            hide_index=True, width="stretch",
        )
        st.caption(
            "SIBAL fabric spreads at a steady 2.1 then 2.6 mm/s in "
            "microgravity. At 1 g the column reads *Acceleratory*: there is no "
            "stable rate to quote. Buoyancy runs away with the flame on Earth; "
            "in weightlessness it advances evenly."
        )

        st.divider()
        fields = load_fields()
        compare = fields[
            fields["variable"].isin(["smoke", "vort", "u", "numden"])
            & fields["thermophoresis"]
        ].pivot_table(index="variable_label", columns="gravity", values="max").round(2)
        st.dataframe(compare, width="stretch")
        st.caption(
            "Vorticity drops from 56.8 to 35.2 and axial velocity from 3.4 to "
            "2.4: that is buoyancy disappearing. Smoke fraction goes the other "
            "way. Simulated cases, normalised units."
        )
        page = threejs_page()
        if page:
            with st.expander("See the plume in three dimensions"):
                components.html(page, height=600, scrolling=False)

    # --- recherche -----------------------------------------------------------
    with tabs[3]:
        engine = report_search()
        if engine is None:
            st.warning(
                "The corpus is not built. Run `py scripts/extract_pdf_text.py`."
            )
        else:
            st.markdown("#### Search NASA's reports")
            st.caption(
                f"{len(engine):,} passages from "
                f"{engine.corpus['document'].nunique()} reports. Every result "
                "carries its document and page: **the search answers only what "
                "it can cite**, and stays silent when it finds nothing."
            )
            question = st.text_input(
                "Question", key="deep_query",
                placeholder="Ask in English; the reports are in English",
                label_visibility="collapsed",
            )
            if question:
                hits = engine.search(question, limit=6)
                if not hits:
                    st.info(
                        "Nothing close enough. The search compares words, not "
                        "ideas."
                    )
                for hit in hits:
                    with st.container(border=True):
                        head, score = st.columns([4, 1])
                        with head:
                            st.markdown(f"**{hit.citation}**")
                        with score:
                            st.markdown(
                                f"<div style='text-align:right;color:#8a93a0'>"
                                f"{hit.score:.3f}</div>", unsafe_allow_html=True)
                        if hit.terms:
                            st.caption("shared terms: " + ", ".join(hit.terms))
                        else:
                            st.caption(":orange[no shared word]: semantic match")
                        st.markdown(f"> {hit.text}")
            with st.expander("How the search works, and where it fails"):
                st.caption(
                    "Word matching alone is blind to rephrasing: on eight "
                    "paraphrased questions it found the right investigation 3 "
                    "times. A dimension reduction finds it 6 times, but on its "
                    "own it loses the right to stay silent: it scored 0.84 on "
                    "*what is the capital of Australia*, higher than most real "
                    "questions."
                )
                st.caption(
                    "So the two run in series: word matching holds the gate, "
                    "the reduction does the ranking. A question must share at "
                    "least two substantial words with the corpus. That keeps 8 "
                    "domain questions out of 8 and turns away 6 off-topic ones "
                    "out of 7."
                )
                st.caption(
                    "**The limit is real.** *stock market prices today* still "
                    "gets through, both its words existing in the corpus. No "
                    "lexical statistic separates off-topic perfectly, which is "
                    "why shared terms are shown beside every result."
                )
                st.dataframe(
                    engine.coverage(), hide_index=True, width="stretch"
                )

    # --- choix des modeles ----------------------------------------------------
    with tabs[4]:
        st.markdown("#### One model per dataset, chosen by measurement")
        st.caption(
            "The plan started with a rule: at these row counts, prefer a linear "
            "model, an ensemble would memorise noise. Measured dataset by "
            "dataset, the rule holds for some and fails for others."
        )
        st.dataframe(
            pd.DataFrame([
                {"dataset": "PSI-69 suppression", "rows": 206,
                 "linear": "0.753", "tree or forest": "unlimited tree 0.678",
                 "kept": "linear"},
                {"dataset": "PSI-39 cool flames", "rows": 144,
                 "linear": "0.771 (AUC 0.913)", "tree or forest": "forest 0.791",
                 "kept": "linear"},
                {"dataset": "PSI-159 sustainment", "rows": 272,
                 "linear": "0.722", "tree or forest": "forest depth 6, 0.840",
                 "kept": "forest"},
                {"dataset": "PSI-101 detection", "rows": 129,
                 "linear": "0.676", "tree or forest": "forest 0.653",
                 "kept": "linear"},
                {"dataset": "PSI-107 soot", "rows": 70,
                 "linear": "0.540", "tree or forest": "forest 0.514",
                 "kept": "linear"},
                {"dataset": "PSI-142 ground", "rows": 95,
                 "linear": "-0.102", "tree or forest": "forest 0.547",
                 "kept": "forest"},
            ]),
            hide_index=True, width="stretch",
        )
        st.caption(
            "On PSI-142 the linear model does **worse than answering the mean**. "
            "On PSI-159 the forest wins by twelve points, and the reason is "
            "physical: fitting each burner configuration separately shows the "
            "pressure coefficient flipping sign between them. No general rule "
            "survives these six rows."
        )

        st.divider()
        st.markdown("**What the dashboard can and cannot do**")
        st.dataframe(coverage_summary(), hide_index=True, width="stretch")
        st.caption(
            "Features use only what is known **before ignition**. Extinction "
            "diameter, burn time and burning rate are measured during or after: "
            "feeding them in would predict extinction from the evidence that it "
            "already happened."
        )
