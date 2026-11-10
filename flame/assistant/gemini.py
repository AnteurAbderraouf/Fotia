"""La conversation avec Gemini, sur l'offre gratuite de Google.

LA CLE. Lue dans st.secrets["GEMINI_API_KEY"] (Streamlit Community Cloud :
Settings > Secrets ; en local : .streamlit/secrets.toml, hors git), sinon dans
la variable d'environnement GEMINI_API_KEY ou GOOGLE_API_KEY. Sans cle,
l'assistant se dit hors ligne ; le reste de la console marche.

LE MODELE. L'offre gratuite change souvent (modeles retires, quotas
resserres). On n'en code donc aucun en dur : `candidates()` demande a l'API
les modeles Flash que cette cle peut utiliser, les plus recents d'abord, la
version « lite » apres la version pleine. Si un modele refuse (quota epuise,
modele retire), on passe au suivant. st.secrets["GEMINI_MODEL"] force un
modele precis.

LA BOUCLE. Appel de fonctions manuel : le modele demande un outil, on
l'execute (tools.run), on lui renvoie le resultat, jusqu'a une reponse en
texte. Au plus MAX_TOOL_ROUNDS tours, pour qu'une question ne puisse pas
vider le quota du jour.

L'HISTORIQUE. Gemini ne garde rien : chaque question renvoie la
conversation. On ne garde que les MAX_TURNS derniers echanges, en texte
seulement ; les resultats d'outils d'une question passee ne sont pas
renvoyes (le modele rappellera l'outil s'il en a besoin, avec les reglages
du moment).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from flame.assistant import tools
from flame.assistant.knowledge import system_prompt

MAX_TOOL_ROUNDS = 6
MAX_TURNS = 8           # echanges gardes dans l'historique envoye
MAX_QUESTION_CHARS = 1200
MAX_OUTPUT_TOKENS = 1200
CALL_TIMEOUT_MS = 25_000
MAX_MODELS_TRIED = 3    # au-dela, le visiteur a deja trop attendu


class Offline(Exception):
    """Pas de cle, pas de SDK ou aucun modele disponible : message pour l'ecran."""


@dataclass
class Reply:
    text: str
    model: str
    tools_used: list[str] = field(default_factory=list)


def api_key() -> str | None:
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return str(key)
    except Exception:  # pas de secrets.toml : st.secrets leve
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def forced_model() -> str | None:
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_MODEL") or os.environ.get("GEMINI_MODEL")
    except Exception:
        return os.environ.get("GEMINI_MODEL")


def _client():
    key = api_key()
    if not key:
        raise Offline("The assistant needs a Gemini API key (free at aistudio.google.com).")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise Offline("The google-genai package is not installed.") from exc
    # Une seule tentative par modele, CALL_TIMEOUT_MS au plus : le SDK
    # reessaierait sinon longtemps sur un modele sature, et le visiteur
    # attendrait devant un spinner. On passe plutot au modele suivant.
    return genai.Client(api_key=key, http_options=types.HttpOptions(
        timeout=CALL_TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1)))


# --- Choix du modele ------------------------------------------------------------

_FLASH = re.compile(r"^(?:models/)?gemini-(\d+(?:\.\d+)?)-flash(-lite)?$")


def _rank(name: str):
    # « lite » d'abord : mesure du 2026-11-09 avec la cle du projet,
    # gemini-3.5-flash-lite repond en 1 s, gemini-3.8-flash depassait 28 s
    # (504). Une conversation attend une reponse rapide ; le Flash complet
    # reste en secours.
    m = _FLASH.match(name)
    version, lite = float(m.group(1)), bool(m.group(2))
    return (not lite, -version)


def candidates(client) -> list[str]:
    """Modeles Flash stables utilisables, les meilleurs d'abord."""
    forced = forced_model()
    if forced:
        return [forced]
    names = []
    for model in client.models.list():
        name = getattr(model, "name", "") or ""
        actions = getattr(model, "supported_actions", None) or []
        if _FLASH.match(name) and (not actions or "generateContent" in actions):
            names.append(name.removeprefix("models/"))
    if not names:
        raise Offline("No Gemini Flash model is available for this API key.")
    return sorted(set(names), key=_rank)


# --- Outils au format Gemini ----------------------------------------------------------

def _declarations(types):
    decls = []
    for name, spec in tools.TOOLS.items():
        decls.append(types.FunctionDeclaration(
            name=name,
            description=spec["description"],
            parameters_json_schema=spec["parameters"],
        ))
    return [types.Tool(function_declarations=decls)]


# --- La boucle ---------------------------------------------------------------------

def _history(types, turns: list[dict]):
    out = []
    for t in turns[-2 * MAX_TURNS:]:
        role = "model" if t["role"] == "assistant" else "user"
        out.append(types.Content(role=role, parts=[types.Part(text=t["text"])]))
    return out


def ask(question: str, turns: list[dict], current: dict) -> Reply:
    """Une question du visiteur -> une reponse. `turns` = [{role, text}, ...]
    deja echanges ; `current` = le reglage de la console."""
    from google.genai import errors, types

    question = question.strip()[:MAX_QUESTION_CHARS]
    client = _client()
    config = types.GenerateContentConfig(
        system_instruction=system_prompt(),
        tools=_declarations(types),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.2,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    last_error = None
    for model in candidates(client)[:MAX_MODELS_TRIED]:
        contents = _history(types, turns)
        contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
        used: list[str] = []
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                response = client.models.generate_content(
                    model=model, contents=contents, config=config)
                calls = response.function_calls or []
                if not calls:
                    text = (response.text or "").strip()
                    if not text:
                        text = "I could not produce an answer. Please rephrase the question."
                    return Reply(text, model, used)
                contents.append(response.candidates[0].content)
                parts = []
                for call in calls:
                    used.append(call.name)
                    result = tools.run(call.name, dict(call.args or {}), current)
                    parts.append(types.Part.from_function_response(
                        name=call.name,
                        response={"result": json.loads(json.dumps(result, default=str))},
                    ))
                contents.append(types.Content(role="user", parts=parts))
            return Reply("This question needed too many lookups. Please ask "
                         "something narrower.", model, used)
        except errors.APIError as exc:
            # 429 : quota de ce modele epuise ; 404 : modele retire ou refuse ;
            # 500/503/504 : modele sature. On essaie le suivant. Le reste (cle
            # invalide...) remonte.
            if getattr(exc, "code", None) in (404, 429, 500, 503, 504):
                last_error = exc
                continue
            if getattr(exc, "code", None) in (400, 401, 403) and "API key" in str(exc):
                raise Offline("The Gemini API key was refused.") from exc
            raise
        except Exception as exc:
            # Delai depasse cote client (httpx2) : meme traitement.
            if "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower():
                last_error = exc
                continue
            raise
    if last_error is not None and getattr(last_error, "code", None) == 429:
        raise Offline("The free Gemini quota is used up for now. Try again later.")
    raise Offline("No Gemini model answered. Try again later.")
