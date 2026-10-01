from haan.lexicon import LABELS, LANGS, PHRASES, SPEAKERS, tts_jobs


def test_every_language_has_both_labels_and_hard_cases():
    for lang in LANGS:
        for label in LABELS:
            items = PHRASES[lang][label]
            assert len(items) >= 10
            assert all(text.strip() for text, _, _ in items)
        kinds = {k for label in LABELS for _, _, k in PHRASES[lang][label]}
        assert kinds == {"easy", "hard"}


def test_no_phrase_has_both_labels():
    for lang in LANGS:
        bc = {t for t, _, _ in PHRASES[lang]["backchannel"]}
        it = {t for t, _, _ in PHRASES[lang]["interrupt"]}
        assert not bc & it


def test_jobs_have_unique_ids_and_known_speakers():
    jobs = tts_jobs()
    assert len({j["id"] for j in jobs}) == len(jobs)
    assert all(j["speaker"] in SPEAKERS[j["lang"]] for j in jobs)
    assert tts_jobs() == jobs  # deterministic


def test_transcript_baselines():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from baseline_asr import classify

    assert classify("Mm-hmm.")["keywords"] == "backchannel"
    assert classify("हाँ जी")["keywords"] == "backchannel"
    assert classify("okay but wait")["keywords"] == "interrupt"
    assert classify("okay but wait")["min_words"] == "interrupt"
    assert classify("wait")["min_words"] == "backchannel"  # why word counts fail on short interruptions
    assert classify("")["keywords"] == "interrupt"
