# haan

Turn-taking for voice agents on **phone calls**.

Voice agents decide when the caller has finished speaking with end-of-turn
models such as [Smart Turn](https://github.com/pipecat-ai/smart-turn). These
models are trained and benchmarked on 16 kHz audio. Phone calls are 8 kHz
G.711 mu-law. haan measures what that costs and fine-tunes the fix. It also
adds a backchannel detector, so a caller saying "haan", "hmm" or "avunu" does
not stop the agent mid-sentence.

## Results

**Smart Turn v3.2, fine-tuned for one epoch on 39k clips, full official test
set (31,527 clips, 23 languages), PSTN channel:**

| | control (normal audio) | telephony (+ phone-audio augmentation) |
|---|---|---|
| Accuracy, phone line | 91.52% | **92.78%** |
| Callers cut off (FPR), phone line | 11.09% | **8.64%** |
| Accuracy lost vs normal audio | 2.01 pts (95% CI 1.77–2.28) | **0.85 pts** (95% CI 0.62–1.05) |
| Accuracy, normal 16 kHz audio | 93.53% | 93.63% |

- **58% less accuracy lost on phone audio**, and **22% fewer cut-off callers.**
- On phone-line audio, accuracy improved in **23 of 23 languages** and FPR fell in 22 of 23.
- Normal-audio accuracy is unchanged overall (+0.1). Per language it is mixed: 14 up, 7 down.
- Both arms start from the same weights and use the same data, steps, learning rate and seed. Only the channel mix differs, so the gain comes from the augmentation, not from the extra training.

The languages hit hardest by phone audio are Indic:

| Language (phone line) | Accuracy, control → telephony | FPR, control → telephony |
|---|---|---|
| Hindi | 90.2 → **92.2** | 13.3 → **10.2** |
| Marathi | 83.5 → **85.1** | 22.6 → **19.7** |
| Bengali | 82.2 → **83.6** | 19.8 → **17.7** |
| English | 93.1 → **94.0** | 9.3 → **7.5** |

Full tables: [`results/report_control.md`](results/report_control.md),
[`results/report_telephony.md`](results/report_telephony.md).

**The original v3.2 on phone audio.** This is the deployed CPU INT8 model,
scored on the first 12,612 test clips:

| Channel | Accuracy | FPR |
|---|---|---|
| wideband 16 kHz | 92.41% | 9.86% |
| 8 kHz mu-law | 90.68% | 15.48% |
| PSTN (mu-law + 300–3400 Hz) | 88.92% | 18.05% |

Accuracy drops by 3.0–4.0 points on PSTN (95% CI, paired bootstrap), and
callers are cut off about 1.8× as often. Hindi FPR goes from 15.1% to 28.6%.
See [`results/report_original_int8_subset.md`](results/report_original_int8_subset.md).
This addresses [pipecat-ai/smart-turn#42](https://github.com/pipecat-ai/smart-turn/issues/42).

> FPR here is the share of *unfinished* turns the model calls finished
> (FP / (FP + TN)). Daily's benchmark reports divide by all clips, so their
> FPR numbers look smaller. Accuracy is directly comparable.

## How it works

```
16 kHz clip ─► resample to 8 kHz ─► [300–3400 Hz band] ─► 8-bit mu-law ─► decode ─► 16 kHz ─► model
              (haan/telephony.py: "narrowband" = without the band filter, "pstn" = with it)
```

1. **Telephone channel simulation** (`haan/telephony.py`): G.711-style 8-bit
   mu-law, an 8 kHz round trip, and the PSTN voice band. Tests check that the
   4–8 kHz band and low rumble are removed.
2. **Recovering the PyTorch weights** (`haan/convert.py`): Daily publishes v3.2
   only as ONNX. The FP32 file stores Linear weights as transposed MatMul
   constants named `val_N`. Each node's `pkg.torch.onnx.name_scopes` metadata
   gives the module path, so every weight maps back to its PyTorch name. The
   rebuilt model matches the ONNX output to 6e-8.
3. **Fine-tuning** (`scripts/train.py`, `colab/train_haan.ipynb`): 12 of 83
   training shards (~39k clips) spread across the set, 1 epoch, AdamW at
   lr 1e-5 with cosine decay and warmup, batch 64, fp16 on a T4. The telephony
   arm sends each clip through wideband, narrowband or PSTN with equal
   probability. Phone channels also get a random level of −12 to +3 dB,
   because mu-law error depends on level.
4. **Evaluation** (`scripts/bench_telephony.py`, `scripts/report_telephony.py`):
   every test clip is scored under all three channels, with per-language and
   per-dataset tables and paired bootstrap CIs.

## Backchannel vs interruption (code ready, not yet trained)

[livekit/agents#6033](https://github.com/livekit/agents/issues/6033) asks for a
self-hosted model that tells backchannels ("haa, avunu, hmm") from real
interruptions ("wait, stop"). LiveKit's model runs only in their cloud.

What is built and tested:
- `haan/backchannel.py`: a small attention-pooling head on the **frozen**
  Smart Turn encoder. One encoder pass gives `(p_complete, p_backchannel)`.
  Turn detection is unchanged by construction, and a test checks it.
- `haan/lexicon.py`: backchannel and interruption phrases in English, Hindi
  and Telugu, in native script. It includes "hard" cases such as
  "okay but wait" and "sare, kaani aagandi", where an acknowledgement word
  leads into an interruption.
- `scripts/gen_tts.py` + `colab/gen_backchannel_tts.ipynb`: 1,562 training
  clips from Indic Parler-TTS (13 voices, varied styles).
- Splits hold out one speaker per language and every fifth phrase.
- `scripts/record_clips.py`: records a **real-speech** test set. Its prompts
  include wording the TTS never saw.
- `scripts/baseline_asr.py`: transcript baselines to beat. These are a keyword
  list and a min-words rule on Whisper large-v3-turbo transcripts, plus
  VAD-only (always interrupt), with ASR latency recorded.

Not done yet: generating the TTS clips, training the head, recording the real
test set, and the comparison against the baselines.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q

# original model on telephone audio (CPU; about 4 min per 3k-clip shard on an M3)
.venv/bin/python -c "from huggingface_hub import hf_hub_download, snapshot_download; \
  hf_hub_download('pipecat-ai/smart-turn-v3','smart-turn-v3.2-cpu.onnx',local_dir='models'); \
  snapshot_download('pipecat-ai/smart-turn-data-v3.2-test',repo_type='dataset',local_dir='data/st-v3.2-test')"
.venv/bin/python scripts/bench_telephony.py
.venv/bin/python scripts/report_telephony.py
```

Fine-tuning both arms and scoring them: open `colab/train_haan.ipynb` on a T4
(about 1.5 h). Outputs go to Google Drive.

## Limitations

- **The phone channel is simulated** (codec, resampling, band filter). It has
  no packet loss, jitter, echo, carrier noise suppression or other codecs
  (Opus, AMR). There are no recordings from real phone lines yet.
- **One seed per arm.** The control-vs-telephony difference is clear in every
  language, but there is no per-clip paired test between the two arms yet.
  The CIs above compare each arm with its own wideband score.
- **The original-model numbers are not like-for-like with the arms.** They use
  the INT8 CPU model on 40% of the test set; the arms use FP32 on all of it.
  The full FP32 original run is a cell in the Colab notebook but has not been
  run yet.
- **The test set is Daily's**, and much of it is synthetic (TTS). Its
  speakers may overlap with the training shards in the same way they do
  upstream.
- **No fine-tuned weights are published yet**, and there is no INT8 export or
  CPU latency measurement of the fine-tuned model.
- **The backchannel head is untrained** (see above).

## Credits and licences

- Smart Turn model, code and data: [Daily / pipecat-ai](https://github.com/pipecat-ai/smart-turn)
  (BSD-2-Clause, data CC-BY-4.0). `haan/model.py` follows its architecture so
  the published weights load directly.
- Encoder: OpenAI Whisper-tiny. TTS: [AI4Bharat Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts) (Apache-2.0).
- haan: BSD-2-Clause, see [LICENSE](LICENSE).

Built by Sri Charan Sripadi ·
[LinkedIn](https://linkedin.com/in/sri-charan-sripadi-063146265) ·
[GitHub](https://github.com/Charansripadi)
