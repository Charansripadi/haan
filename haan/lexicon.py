"""Phrases a caller says while the agent is talking.

label "backchannel": acknowledgement, the agent should keep talking.
label "interrupt":   the caller wants the floor, the agent should stop.

"hard" phrases are the cases keyword lists get wrong: a backchannel word that
leads into a real interruption ("okay, but wait"), or a long acknowledgement.
Native script is what the TTS reads; the romanisation is for review only.
"""

from __future__ import annotations

# (text, romanisation, kind) per language and label
PHRASES: dict[str, dict[str, list[tuple[str, str, str]]]] = {
    "eng": {
        "backchannel": [
            ("hmm", "", "easy"), ("mm-hmm", "", "easy"), ("uh-huh", "", "easy"), ("yeah", "", "easy"),
            ("okay", "", "easy"), ("right", "", "easy"), ("I see", "", "easy"), ("sure", "", "easy"),
            ("got it", "", "easy"), ("yes", "", "easy"), ("alright", "", "easy"), ("oh okay", "", "easy"),
            ("yeah yeah", "", "easy"), ("okay okay", "", "easy"), ("that makes sense", "", "hard"),
            ("yes, I'm listening", "", "hard"), ("okay, go on", "", "hard"), ("right, right", "", "easy"),
        ],
        "interrupt": [
            ("wait", "", "easy"), ("stop", "", "easy"), ("hold on", "", "easy"), ("actually", "", "easy"),
            ("no no", "", "easy"), ("sorry, can I ask something", "", "easy"), ("let me explain", "", "easy"),
            ("that's not what I said", "", "easy"), ("one second", "", "easy"), ("excuse me", "", "easy"),
            ("can you repeat that", "", "easy"), ("I want to talk to a person", "", "easy"),
            ("okay but wait", "", "hard"), ("yeah, no, actually", "", "hard"), ("hmm, but what about my refund", "", "hard"),
            ("right, so how much is it", "", "hard"), ("okay, stop", "", "hard"), ("yes, but I already paid", "", "hard"),
        ],
    },
    "hin": {
        "backchannel": [
            ("हाँ", "haan", "easy"), ("हाँ जी", "haan ji", "easy"), ("अच्छा", "achha", "easy"),
            ("ठीक है", "theek hai", "easy"), ("हम्म", "hmm", "easy"), ("जी", "ji", "easy"),
            ("हाँ हाँ", "haan haan", "easy"), ("सही है", "sahi hai", "easy"), ("ओके", "okay", "easy"),
            ("अच्छा अच्छा", "achha achha", "easy"), ("जी हाँ", "ji haan", "easy"),
            ("हाँ, मैं सुन रहा हूँ", "haan, main sun raha hoon", "hard"), ("ठीक है, बोलिए", "theek hai, boliye", "hard"),
        ],
        "interrupt": [
            ("रुको", "ruko", "easy"), ("एक मिनट", "ek minute", "easy"), ("नहीं नहीं", "nahin nahin", "easy"),
            ("सुनिए", "suniye", "easy"), ("रुकिए", "rukiye", "easy"), ("मेरी बात सुनिए", "meri baat suniye", "easy"),
            ("ये गलत है", "ye galat hai", "easy"), ("फिर से बोलिए", "phir se boliye", "easy"),
            ("हाँ लेकिन रुको", "haan lekin ruko", "hard"), ("अच्छा, पर मेरा पैसा कब आएगा", "achha, par mera paisa kab aayega", "hard"),
            ("ठीक है, पर एक सवाल है", "theek hai, par ek sawaal hai", "hard"), ("हम्म, नहीं, ऐसा नहीं है", "hmm, nahin, aisa nahin hai", "hard"),
        ],
    },
    "tel": {
        "backchannel": [
            ("అవును", "avunu", "easy"), ("హా", "haa", "easy"), ("సరే", "sare", "easy"), ("ఓకే అండి", "okay andi", "easy"),
            ("హ్మ్", "hmm", "easy"), ("అలాగే", "alaage", "easy"), ("అవునండి", "avunandi", "easy"),
            ("సరే సరే", "sare sare", "easy"), ("హా హా", "haa haa", "easy"), ("అర్థమైంది", "arthamaindi", "easy"),
            ("చెప్పండి, వింటున్నాను", "cheppandi, vintunnaanu", "hard"), ("అవును, అలాగే", "avunu, alaage", "hard"),
        ],
        "interrupt": [
            ("ఆగండి", "aagandi", "easy"), ("ఒక్క నిమిషం", "okka nimisham", "easy"), ("కాదు కాదు", "kaadu kaadu", "easy"),
            ("వినండి", "vinandi", "easy"), ("ఆపండి", "aapandi", "easy"), ("మళ్ళీ చెప్పండి", "malli cheppandi", "easy"),
            ("అది తప్పు", "adi tappu", "easy"), ("నా మాట వినండి", "naa maata vinandi", "easy"),
            ("సరే, కానీ ఆగండి", "sare, kaani aagandi", "hard"), ("అవును, కానీ నా డబ్బులు ఎప్పుడు వస్తాయి", "avunu, kaani naa dabbulu eppudu vastaayi", "hard"),
            ("హా, కానీ ఒక ప్రశ్న", "haa, kaani oka prashna", "hard"),
        ],
    },
}

LABELS = ("backchannel", "interrupt")
LANGS = tuple(PHRASES)

# Indic Parler-TTS recommended speakers, plus unnamed descriptions for variety
SPEAKERS = {
    "eng": ["Thoma", "Mary", "Swapna", "Dinesh", "Meera", "Jatin"],
    "hin": ["Rohit", "Divya", "Aman", "Rani"],
    "tel": ["Prakash", "Lalitha", "Kiran"],
}
STYLES = [
    "speaks casually at a moderate pace with a close-sounding, clear recording",
    "speaks quickly and softly, slightly distant, with a little background noise",
    "speaks slowly with a low pitch in a quiet room",
    "speaks with an animated, slightly high-pitched voice and very clear audio",
    "speaks in a flat, monotone voice with moderate background noise",
    "speaks firmly and a bit loudly, close to the microphone",
]


REPS = {"eng": 3, "hin": 5, "tel": 6}  # fewer speakers per Indic language, so more styles each


def tts_jobs(reps: dict[str, int] | None = None, seed: int = 0) -> list[dict]:
    """Every phrase x speaker x reps[lang] random styles, with stable ids."""
    reps = reps or REPS
    import random

    rng = random.Random(seed)
    jobs = []
    for lang, by_label in PHRASES.items():
        for label, items in by_label.items():
            for pi, (text, roman, kind) in enumerate(items):
                for spk in SPEAKERS[lang]:
                    for r in range(reps[lang]):
                        style = rng.choice(STYLES)
                        jobs.append({
                            "id": f"{lang}-{label[:2]}-{pi:02d}-{spk}-{r}",
                            "lang": lang, "label": label, "kind": kind,
                            "text": text, "roman": roman or text,
                            "speaker": spk, "description": f"{spk} {style}.",
                        })
    return jobs
