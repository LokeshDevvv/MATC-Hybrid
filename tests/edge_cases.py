"""Edge-case test suite for the text analyzer.

Hits the running /analyze endpoint on http://localhost:8001 with a battery
of tricky inputs and prints a compact report showing what each model said.
"""

import json
import sys
import time
import urllib.request

API = "http://127.0.0.1:8001/analyze"


def call(text: str, timeout: int = 90):
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        API, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return data, time.time() - t0
    except Exception as e:
        return {"error": str(e)}, time.time() - t0


def fmt(d):
    """One-line summary of analysis output."""
    if "error" in d:
        return f"ERROR: {d['error']}"
    s = (d.get("sentiment") or {})
    e = (d.get("emotions") or [{}])[0]
    t = (d.get("topic") or {})
    i = (d.get("intent") or {})
    u = (d.get("urgency") or {})
    sa = (d.get("sarcasm") or {})
    tox = (d.get("toxicity") or {})
    lang = (d.get("language") or {})
    ents = d.get("entities") or []
    flags = []
    if sa.get("is_sarcastic"): flags.append("SARC")
    if tox.get("is_toxic"): flags.append("TOXIC")
    return (
        f"lang={lang.get('language','-')}({lang.get('score',0):.2f}) | "
        f"sent={s.get('label','-')}({s.get('score',0):.2f}) | "
        f"emo={e.get('label','-')}({e.get('score',0):.2f}) | "
        f"topic={t.get('label','-')} | "
        f"intent={i.get('label','-')} | "
        f"urg={u.get('label','-')} | "
        f"flags={','.join(flags) if flags else '-'} | "
        f"ents={len(ents)}"
    )


# ─── Test cases ─────────────────────────────────────────────────────────
CASES = [
    # ── Boundary / empty / minimal
    ("EMPTY",                ""),
    ("SINGLE_SPACE",         " "),
    ("SINGLE_CHAR",          "a"),
    ("SINGLE_WORD",          "hello"),
    ("PUNCTUATION_ONLY",     "!!!???..."),
    ("NUMBERS_ONLY",         "123 456 789"),
    ("EMOJI_ONLY",           "😀😀😀🎉"),

    # ── Length stress
    ("VERY_LONG", "This is a great sentence. " * 50),
    ("HUGE_PARAGRAPH", "I love this product. It's amazing. The team built something special. " * 30),

    # ── Sentiment edge cases
    ("DOUBLE_NEG",           "It's not that I don't like it."),
    ("MIXED_SENT",           "I love the food but the service was terrible."),
    ("UNDERSTATEMENT",       "It wasn't bad, I suppose."),
    ("SUBTLE_PRAISE",        "He's, you know, decent at his job."),
    ("ALL_CAPS_RAGE",        "I HATE THIS SO MUCH WHY DOES IT KEEP HAPPENING"),

    # ── Sarcasm edge cases
    ("OBVIOUS_SARC",         "Oh wonderful, another bug. Just what I needed."),
    ("DEADPAN_SARC",         "I just love when the wifi goes down during a deadline."),
    ("DRY_SINCERE",          "The presentation was professional and informative."),
    ("AMBIGUOUS",            "Sure, that'll work."),

    # ── Toxic content
    ("INSULT",               "You are an absolute moron and I hate you."),
    ("HATEFUL",              "Get out of my country, you don't belong here."),
    ("MILD_FRUSTRATION",     "Ugh this is so annoying, what is wrong with you people."),
    ("REPORTED_SPEECH",      "She told him he was being an idiot."),

    # ── Multilingual
    ("FRENCH",               "Bonjour, je voudrais une baguette s'il vous plaît."),
    ("SPANISH",              "Hola, ¿cómo estás? Espero que tengas un buen día."),
    ("HINDI",                "नमस्ते, आज कैसा दिन रहा?"),
    ("TAMIL",                "வணக்கம், இன்று நான் மிகவும் மகிழ்ச்சியாக இருக்கிறேன்."),
    ("ARABIC",               "مرحبا، كيف حالك اليوم؟"),
    ("CHINESE",              "今天天气真好，我很开心。"),
    ("JAPANESE",             "今日は素晴らしい一日でした。"),
    ("CODE_SWITCH",          "Hello नमस्ते, today मेरा birthday है!"),

    # ── Entities edge cases
    ("MONEY_FORMATS",        "We spent $5 billion, €3 million, and ₹5,000 yesterday."),
    ("DATE_FORMATS",         "On 2024-05-06 at 3pm, or maybe May 6th, or just tomorrow."),
    ("PERCENT_AND_NUM",      "Sales rose 40% to $5B, beating estimates of 35%."),
    ("PRODUCT_NAMES",        "I bought an iPhone 15 Pro, a MacBook Air, and a Tesla Model 3."),
    ("HISTORICAL_FIGURE",    "Albert Einstein wrote a letter to Niels Bohr in 1927."),
    ("AMBIGUOUS_NAME",       "Apple bit me when I was a kid."),  # Apple here = fruit, not org

    # ── Format / structure
    ("URL_HEAVY",            "Check out https://example.com/path?q=1 for more info."),
    ("EMAIL",                "Send it to john.doe@example.com please."),
    ("MENTIONS",             "@alice please look at this. cc @bob #urgent"),
    ("HASHTAGS",             "Just shipped! #python #ai #ml"),
    ("CODE",                 "def f(x): return x ** 2 # squares the input"),
    ("HTML",                 "<p>Hello <b>world</b></p>"),
    ("MARKDOWN",             "# Heading\n**bold** and *italic* text"),

    # ── Conversational / informal
    ("TWEET_STYLE",          "lol that's so true bro 💀💀💀 fr fr"),
    ("ABBREV",               "tbh idk wym, lmk asap"),
    ("RUN_ON",               "okay so basically what happened was the thing broke and then everything went wrong and I was like wait what"),
    ("FRAGMENT",             "Awesome."),

    # ── Question / intent edge cases
    ("RHETORICAL",           "Is anyone really surprised that this happened?"),
    ("DIRECT_REQUEST",       "Please send me the report by 5pm today."),
    ("COMMAND",              "Stop doing that immediately."),

    # ── Multi-sentence / mixed
    ("PARAGRAPH_MIX",        "I waited 45 minutes. The waiter was rude. But the dessert was incredible. Honestly might come back."),
    ("APOLOGY_AND_THREAT",   "Sorry for the delay. But if it happens again I'm escalating to your manager."),

    # ── Tricky NER
    ("WORK_OF_ART",          "Have you read War and Peace by Tolstoy?"),
    ("LAW",                  "The First Amendment protects free speech."),
    ("LANGUAGE_NAME",        "She speaks fluent Mandarin and Tamil."),
    ("PERCENT_COMPLEX",      "Approximately 12.5% of respondents agreed."),
]


def main():
    results = []
    print(f"Running {len(CASES)} edge-case tests against {API}\n")
    print(f"{'TAG':<22}  {'TIME':>5}  SUMMARY")
    print("─" * 110)
    for tag, text in CASES:
        d, dt = call(text)
        line = fmt(d)
        results.append({"tag": tag, "text": text, "result": d, "time": dt})
        # truncate output line so terminal stays readable
        print(f"{tag:<22}  {dt:5.1f}s  {line}")

    # Save full results for inspection
    with open("tests/edge_case_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full results to tests/edge_case_results.json")

    # Identify suspicious results
    print("\n\n──── Issues found ────")
    issues = []

    for r in results:
        tag = r["tag"]
        d = r["result"]
        text = r["text"]
        if "error" in d:
            issues.append(f"[{tag}] errored out: {d['error']}")
            continue

        sent = (d.get("sentiment") or {}).get("label")
        sarc = (d.get("sarcasm") or {}).get("is_sarcastic")
        tox = (d.get("toxicity") or {}).get("is_toxic")
        lang = (d.get("language") or {}).get("language")

        # Heuristic checks for specific cases
        if tag == "OBVIOUS_SARC" and not sarc:
            issues.append(f"[{tag}] missed sarcasm")
        if tag == "DEADPAN_SARC" and not sarc:
            issues.append(f"[{tag}] missed deadpan sarcasm")
        if tag == "DRY_SINCERE" and sarc:
            issues.append(f"[{tag}] false-positive sarcasm")
        if tag == "INSULT" and not tox:
            issues.append(f"[{tag}] failed to flag insult")
        if tag == "REPORTED_SPEECH" and tox:
            issues.append(f"[{tag}] false-flagged reported speech")
        if tag == "ALL_CAPS_RAGE" and sent != "negative":
            issues.append(f"[{tag}] sentiment={sent} (expected negative)")
        if tag == "TAMIL" and lang != "ta":
            issues.append(f"[{tag}] lang={lang} (expected ta)")
        if tag == "HINDI" and lang != "hi":
            issues.append(f"[{tag}] lang={lang} (expected hi)")
        if tag == "FRENCH" and lang != "fr":
            issues.append(f"[{tag}] lang={lang} (expected fr)")
        if tag == "MIXED_SENT" and sent == "neutral":
            issues.append(f"[{tag}] could not pick a side: {sent}")
        if tag == "DOUBLE_NEG" and sent == "negative":
            issues.append(f"[{tag}] tripped on double negation: {sent}")

    if not issues:
        print("None — all assertions passed!")
    else:
        for i in issues:
            print(" -", i)


if __name__ == "__main__":
    main()
