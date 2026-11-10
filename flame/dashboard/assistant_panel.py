"""Ask Fotia — la conversation, en fenetre flottante qui suit le visiteur.

POURQUOI FLOTTANTE. Placee sous la console, la conversation obligeait a
descendre tout en bas pour poser une question, puis a remonter pour voir la
flamme. Elle est maintenant fixee en bas a droite de l'ecran (CSS
position: fixed sur un conteneur Streamlit nomme) : elle reste visible
pendant qu'on fait defiler la page, et se replie en un seul bouton.

Widgets Streamlit natifs (st.chat_message, st.chat_input), habilles aux
couleurs de la console par flame/viz/assistant.css. Le reglage courant est lu
dans st.session_state[STATE], le meme que celui de la console : « pourquoi ce
badge ? » parle de ce que le visiteur voit.

UNE LIMITE PAR VISITEUR. L'offre gratuite de Gemini a un quota par jour pour
toute la cle. MAX_QUESTIONS par session evite qu'un seul visiteur le vide.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from flame.assistant import gemini
from flame.dashboard.console import STATE, initial

HISTORY = "fotia_chat"
PENDING = "fotia_chat_pending"
OPEN = "fotia_chat_open"
MAX_QUESTIONS = 25
MESSAGES_HEIGHT = 360  # px, la zone qui defile dans la fenetre

SUGGESTIONS = [
    "Why does my setting get this confidence badge?",
    "What is the burning rate constant K?",
    "Would heptane burn out in Orion's atmosphere?",
    "What happened in the Apollo 1 fire?",
]

CSS = Path(__file__).resolve().parents[1] / "viz" / "assistant.css"


def _ask(question: str) -> None:
    st.session_state[PENDING] = question


def _toggle() -> None:
    st.session_state[OPEN] = not st.session_state.get(OPEN, False)


# Icones au trait, sans les visages colores par defaut de Streamlit.
AVATAR = {"user": ":material/person:", "assistant": ":material/local_fire_department:"}


def _message(turn: dict) -> None:
    role = "assistant" if turn["role"] == "assistant" else "user"
    with st.chat_message(role, avatar=AVATAR[role]):
        st.markdown(turn["text"])
        if turn.get("meta"):
            st.html(f'<div class="ask-meta">{turn["meta"]}</div>')


def render() -> None:
    st.html(f"<style>{CSS.read_text(encoding='utf-8')}</style>")

    if not st.session_state.get(OPEN, False):
        with st.container(key="ask_fab"):
            st.button("Ask Fotia", icon=":material/forum:", on_click=_toggle,
                      help="Questions about this dashboard and flames in microgravity")
        return

    history: list[dict] = st.session_state.setdefault(HISTORY, [])
    current = st.session_state.get(STATE) or initial()

    with st.container(key="ask_float"):
        with st.container(key="ask_bar", horizontal=True, vertical_alignment="center"):
            st.html('<div class="ask-title">Ask Fotia</div>')
            st.button("", icon=":material/close:", key="ask_close", on_click=_toggle,
                      help="Close")

        # autoscroll : la zone suit le dernier message, comme une messagerie.
        box = st.container(height=MESSAGES_HEIGHT, key="ask_msgs", autoscroll=True)
        with box:
            if not history:
                st.html('<p class="ask-sub">Questions about this dashboard, flames in '
                        'microgravity and spacecraft fire safety. Every number comes '
                        'from Fotia\'s engine and its NASA sources.</p>')
                with st.container(key="ask_suggest"):
                    for i, s in enumerate(SUGGESTIONS):
                        st.button(s, key=f"ask_s{i}", on_click=_ask, args=(s,))
            for turn in history:
                _message(turn)

        asked = sum(1 for t in history if t["role"] == "user")
        if asked >= MAX_QUESTIONS:
            st.caption("Question limit reached for this visit. Reload the page to start over.")
            return

        typed = st.chat_input("Ask a question", max_chars=gemini.MAX_QUESTION_CHARS)

    question = typed or st.session_state.pop(PENDING, None)
    if not question:
        return

    with box:
        _message({"role": "user", "text": question})
        with st.chat_message("assistant", avatar=AVATAR["assistant"]):
            with st.spinner("Checking Fotia's engine and sources"):
                # Les echecs restent affiches mais ne repartent pas a Gemini.
                ok = True
                try:
                    reply = gemini.ask(question, [t for t in history if t.get("ok", True)], current)
                    text = reply.text
                    meta = (f"Gemini {reply.model}"
                            + (f" · used {', '.join(dict.fromkeys(reply.tools_used))}"
                               if reply.tools_used else ""))
                except gemini.Offline as exc:
                    text, meta, ok = f"Assistant offline. {exc}", "", False
                except Exception as exc:  # une panne du service ne casse pas la page
                    text, meta, ok = ("The assistant could not answer just now. Please try again.",
                                      type(exc).__name__, False)
            st.markdown(text)
            if meta:
                st.html(f'<div class="ask-meta">{meta}</div>')

    history.append({"role": "user", "text": question, "ok": ok})
    history.append({"role": "assistant", "text": text, "meta": meta, "ok": ok})
    # Le message vient d'etre ecrit en bas de la zone ; on relance pour que
    # la suggestion disparaisse et que l'historique soit rendu dans l'ordre.
    st.rerun()
