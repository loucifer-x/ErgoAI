import json
import re

import requests

# ── Model selection ───────────────────────────────────────────────────────────
# Try the config file first; fall back to the default if it's missing.
currentmodel = "qwen2.5:0.5b"
"""
try:
    with open("extraconfig.json") as f:
        currentmodel = json.load(f).get("model", _DEFAULT_MODEL)
except (FileNotFoundError, json.JSONDecodeError):
    currentmodel = _DEFAULT_MODEL
"""
"""
# Try phi3:mini if available (faster for classification)
try:
    resp = requests.get("http://localhost:11434/api/tags", timeout=3)
    available = [m["name"] for m in resp.json().get("models", [])]
    if "phi3:mini" in available:
        currentmodel = "phi3:mini"
except Exception:
    pass
"""

# ── Compiled constants ────────────────────────────────────────────────────────
_LABEL_PREFIX_RE = re.compile(r"^(category|subcategory)\s*[:\-]\s*", re.IGNORECASE)
_OLLAMA_URL      = "http://localhost:11434/api/generate"
_MAX_CHARS       = 1500
_REQUEST_TIMEOUT = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_value(text: str) -> str:
    text = _LABEL_PREFIX_RE.sub("", text.strip())
    text = text.replace("'", "").replace('"', "")
    text = text.split(";")[0].split(",")[0].strip()
    return re.sub(r"\s+", "-", text).lower()


def safe_fallback(output: str):
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", output)
    words = [w for w in cleaned.split() if w]
    if len(words) >= 2:
        return words[0].lower(), words[1].lower()
    if len(words) == 1:
        return words[0].lower(), "general"
    return "unknown", "general"


def classify_text(text: str):
    print(text)
    truncated = text[:_MAX_CHARS]
    url = text
    hint = url  # URL is the hint, not the text
    if "wikipedia.org/wiki/" in url:
        hint = url.split("wikipedia.org/wiki/")[-1].replace("_", " ").lower()
        for noise in ["tv series", "film", "anime", "episodes", "list of"]:
            hint = hint.replace(noise, "").strip()

    hint_line = f"URL hint: {hint}\n\n" if hint else ""

    examples = [
        ("wiki tokyo ghoul",                "entertainment | tokyo ghoul"),
        ("wiki attack on titan",            "entertainment | attack on titan"),
        ("wiki tony montana",               "entertainment | scarface"),
        ("wiki tony soprano",               "entertainment | the sopranos"),
        ("wiki list of stand up comedians", "entertainment | comedy"),
        ("nmap port scanning",              "tools | nmap"),
        ("nvd nist cve 2024",               "redteam | cve"),
    ]

    example_block = "\n\n".join(
        f"  URL hint: {h}\n  {c}" for h, c in examples
    )

    prompt = (
        "Classify the text into a category and subcategory.\n\n"
        f"{hint_line}"
        "Rules:\n"
        "- Reply with ONLY two lowercase words separated by a pipe, like: entertainment | comedy\n"
        "- No punctuation except the pipe |\n"
        "- For named works (films, shows, anime, games, books) use the title as the subcategory\n"
        "- For general topics use a descriptive word\n"
        "- Use the URL hint if the text is sparse\n\n"
        "Examples:\n"
        f"{example_block}\n\n"
        f"Text: {truncated}\n\n"
        "Reply:"
    ).strip()
    try:
        response = requests.post(
            _OLLAMA_URL,
            json={
                "model": currentmodel,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 20,
                },
            },
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        output = response.json().get("response", "").strip()
    except Exception:
        return "unknown", "general"

    for line in output.splitlines():
        line = line.strip().strip("`\"'")
        if "|" not in line:
            continue
        left, _, right = line.partition("|")
        cat = clean_value(left)
        sub = clean_value(right)
        # Reject if the model still echoed the placeholder words
        if cat and sub and cat != "category" and sub != "subcategory":
            return cat, sub

    return safe_fallback(output)