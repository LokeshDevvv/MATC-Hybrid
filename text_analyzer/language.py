"""Language detector using facebook/fasttext-language-identification.

Supports ~200 languages including Tamil, Telugu, Kannada, Malayalam, etc.
Output is mapped to ISO 639-1 (2-letter) codes when possible.
"""

import os
import fasttext
from huggingface_hub import hf_hub_download

MODEL_REPO = "facebook/fasttext-language-identification"
MODEL_FILE = "model.bin"


# fasttext labels are like "__label__eng_Latn", "__label__tam_Taml" etc.
# Map common ones to 2-letter ISO codes for the UI.
ISO3_TO_ISO1 = {
    "eng": "en", "fra": "fr", "spa": "es", "deu": "de", "ita": "it", "por": "pt",
    "rus": "ru", "jpn": "ja", "zho": "zh", "kor": "ko", "ara": "ar", "tur": "tr",
    "vie": "vi", "tha": "th", "hin": "hi", "ben": "bn", "tam": "ta", "tel": "te",
    "kan": "kn", "mal": "ml", "guj": "gu", "pan": "pa", "ori": "or", "urd": "ur",
    "nld": "nl", "pol": "pl", "swe": "sv", "fin": "fi", "ell": "el", "heb": "he",
    "ind": "id", "msa": "ms", "ces": "cs", "ron": "ro", "bul": "bg", "hun": "hu",
    "dan": "da", "nor": "no", "ukr": "uk", "fas": "fa", "swa": "sw", "sin": "si",
    "mya": "my", "khm": "km", "lao": "lo",
}


def _parse_label(label: str):
    """fasttext label -> short code. e.g. '__label__eng_Latn' -> 'en'."""
    s = label.replace("__label__", "")
    iso3 = s.split("_")[0]
    return ISO3_TO_ISO1.get(iso3, iso3)


class LanguageDetector:
    def __init__(self, device=-1):
        # Download once and cache
        path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        self.model = fasttext.load_model(path)

    def analyze(self, text: str):
        text = (text or "").replace("\n", " ").strip()
        if not text:
            return {"language": "—", "score": 0.0}
        labels, probs = self.model.predict(text, k=1)
        lang = _parse_label(labels[0])
        return {"language": lang, "score": round(float(probs[0]), 4)}


if __name__ == "__main__":
    d = LanguageDetector()
    samples = [
        "Hello, how are you doing?",
        "Bonjour, comment allez-vous?",
        "Hola, ¿cómo estás?",
        "नमस्ते, आप कैसे हैं?",
        "வணக்கம், எப்படி இருக்கிறீர்கள்?",
        "你好，你今天怎么样？",
        "ನೀವು ಹೇಗಿದ್ದೀರಿ?",
        "آپ کیسے ہیں؟",
    ]
    for s in samples:
        print(f"{s} -> {d.analyze(s)}")
