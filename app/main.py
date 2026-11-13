"""Fotia — flamme en microgravite et securite incendie des vaisseaux.

    py -m streamlit run app/main.py

UNE PAGE, UNE CONSOLE. L'ancienne interface (niveaux 00 a 03, questions,
catalogue, onglets de verification) a ete retiree le 2026-11-09 : elle
empilait trop de panneaux sans fil conducteur. Tout l'ecran est maintenant la
console de flame/dashboard/console.py, lue dans un ordre fixe :

    01 Where   02 The air   03 The flame   04 Prediction   05 Why it matters

et, par-dessus, Ask Fotia : l'assistant (flame/assistant/), en fenetre
flottante en bas a droite, qui repond avec Gemini en tirant chaque chiffre
du moteur.

Ce fichier ne fait que preparer la page : titre, polices du Horizon Design
System de la NASA (Public Sans, DM Mono), fond charbon, chrome Streamlit
masque. Les anciens modules de flame/dashboard restent sur le disque, non
utilises.

L'INTERFACE EST EN ANGLAIS, LE CODE EN FRANCAIS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flame.dashboard.assistant_panel import render as render_assistant  # noqa: E402
from flame.dashboard.console import render  # noqa: E402

# Icone d'onglet : l'embleme du logo sur fond charbon (blanc seul, il
# disparaitrait sur un onglet clair).
ICON = Path(__file__).resolve().parent / "static" / "icon.png"

st.set_page_config(page_title="Fotia", layout="wide", page_icon=str(ICON),
                   initial_sidebar_state="collapsed")

# Polices chargees au niveau du document : une @font-face declaree dans le
# shadow DOM du composant ne serait pas prise en compte par le navigateur.
st.html(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Public%20Sans:wght@400;500;600&family=Saira:ital,wght@1,500;1,600&display=swap">
    <style>
      .stApp, [data-testid="stAppViewContainer"] { background: #222225; }
      [data-testid="stHeader"], [data-testid="stToolbar"],
      [data-testid="stDecoration"], footer { display: none !important; }
      .block-container { padding: 24px 32px 16px !important; max-width: 1680px; }
    </style>
    """
)

render()
render_assistant()
