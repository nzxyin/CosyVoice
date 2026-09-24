# Fun-CosyVoice3-0.5B-2512, ESD emotion conditions: emotion cloning vs. emotion label (generated 2026-09-24)

Both rows are cross-utterance (the target's own recording is never the prompt); n = 1500 (ESD test, 10 speakers x 5 emotions x 30).

- `cross` = **emotion cloning**: prompt = next stem in the target's (speaker, emotion) group, cyclic, via `inference_zero_shot`. Identical pairing to articulatory-tts GH #91's `emotion_ref_mode=cross` (`esd_english_splits/test_cross_ref_pairs.tsv`).
- `cross_label` = **emotion label**: prompt = a Neutral utterance of the same speaker, emotion given as an instruct (`en_plain_neutral`: "Please say a sentence in a very <angry|happy|sad|surprised> tone.", no instruct for Neutral) via `inference_instruct2` (the prompt's speech tokens are not given to the LLM). Jobs 10552443-10552447.
- `Acc cos` in the overall table is NOT comparable across the two rows: `cross` is raw GenAID (2026-09-14 rescore), `cross_label` is centered GenAID (eight-accent vector, the scorer's current default). Accent has no meaning on ESD anyway; the per-emotion accent column is only filled for `cross`.

Paired differences, label minus cloning (same 1500 targets, bootstrap 95% CI):

| metric | all | Angry | Happy | Neutral | Sad | Surprise |
|---|---|---|---|---|---|---|
| emotion cosine | -0.149 [-0.160, -0.138] | -0.088 [-0.108, -0.067] | -0.186 [-0.210, -0.162] | -0.058 [-0.072, -0.043] | -0.239 [-0.263, -0.213] | -0.172 [-0.203, -0.142] |
| speaker cosine | -0.104 [-0.109, -0.098] | -0.085 [-0.097, -0.074] | -0.107 [-0.119, -0.095] | -0.068 [-0.078, -0.058] | -0.174 [-0.186, -0.161] | -0.085 [-0.097, -0.072] |
| UTMOSv2 | -0.030 [-0.055, -0.007] | -0.045 [-0.099, +0.013] | -0.038 [-0.089, +0.014] | -0.121 [-0.169, -0.075] | -0.046 [-0.098, +0.002] | +0.098 [+0.041, +0.155] |

Neutral targets get the same prompt recording in both conditions and no emotion instruct in `cross_label`, so the Neutral column isolates the cost of the instruct inference path itself (the LLM loses the prompt's speech tokens).

Instruct-wording pilot (ESD **val**, 100 utts = 2 per speaker x emotion, job 10552276; emotion2vec+ large cosine / fraction labelled as the target emotion):

| condition | Angry | Happy | Neutral | Sad | Surprise | all | labelled target (all) |
|---|---|---|---|---|---|---|---|
| cloning (`cross`) | 0.956 | 0.687 | 0.933 | 0.717 | 0.554 | 0.769 | 0.56 |
| label, `zh` (repo's 请非常X地说一句话。 template) | 0.744 | 0.544 | 0.940 | 0.520 | 0.301 | 0.610 | 0.35 |
| label, `en` | 0.790 | 0.630 | 0.848 | 0.609 | 0.323 | 0.640 | 0.35 |
| label, `none` (Neutral prompt, no instruct) | 0.546 | 0.423 | 0.917 | 0.488 | 0.224 | 0.520 | 0.20 |

Paired: en - zh on the four non-Neutral emotions +0.061 [+0.033, +0.089]; on Neutral en - none -0.069 [-0.120, -0.020], zh - none +0.023 [-0.005, +0.051]. Hence `en_plain_neutral` (en for the four emotions, no instruct for Neutral).

| prompt | dataset | n (scored/total) | WER% raw | WER% whisper-norm | WER% vs spoken | UTMOSv2 | DNSMOS ovr | DNSMOS p808 | DNSMOS sig | DNSMOS bak | Spk cos | Emo cos | Acc cos | RTF | #pred<0.5s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cross | esd | 1500/1500 | 14.65 | 2.27 | 14.65 | 3.524 ±0.020 | 3.193 ±0.010 | 3.589 ±0.014 | 3.475 ±0.008 | 4.088 ±0.007 | 0.580 ±0.006 | 0.764 ±0.012 | 0.971 ±0.001 | 0.634 | 0 |
| cross_label | esd | 1500/1500 | 12.97 | 2.61 | 12.97 | 3.494 ±0.019 | 3.118 ±0.011 | 3.569 ±0.013 | 3.447 ±0.008 | 3.985 ±0.010 | 0.476 ±0.006 | 0.616 ±0.013 | 0.785 ±0.011 | 0.701 | 0 |

cross/esd by_emotion:
| group | n | WER% raw | WER% whisper-norm | UTMOSv2 | DNSMOS ovr | Spk cos | Emo cos | Acc cos | pred/GT dur | pred->target label | GT->target label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Angry | 300 | 14.77 | 2.73 | 3.545 ±0.047 | 3.209 ±0.023 | 0.578 ±0.011 | 0.891 ±0.015 | 0.972 ±0.003 | 1.05 ±0.03 | 0.83 | 0.98 |
| Happy | 300 | 13.89 | 1.92 | 3.551 ±0.045 | 3.186 ±0.021 | 0.553 ±0.012 | 0.768 ±0.021 | 0.974 ±0.003 | 1.06 ±0.03 | 0.60 | 0.93 |
| Neutral | 300 | 15.01 | 2.28 | 3.596 ±0.045 | 3.204 ±0.021 | 0.623 ±0.011 | 0.912 ±0.011 | 0.973 ±0.004 | 1.09 ±0.03 | 0.96 | 1.00 |
| Sad | 300 | 14.78 | 1.96 | 3.499 ±0.041 | 3.212 ±0.020 | 0.622 ±0.011 | 0.786 ±0.020 | 0.963 ±0.004 | 1.06 ±0.03 | 0.64 | 0.99 |
| Surprise | 300 | 14.79 | 2.46 | 3.429 ±0.047 | 3.152 ±0.026 | 0.524 ±0.012 | 0.464 ±0.029 | 0.973 ±0.003 | 1.09 ±0.03 | 0.19 | 0.95 |

cross/esd by_speaker:
| group | n | WER% raw | WER% whisper-norm | UTMOSv2 | DNSMOS ovr | Spk cos | Emo cos | Acc cos | pred/GT dur | pred->target label | GT->target label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0011 | 150 | 14.42 | 2.83 | 3.387 ±0.063 | 3.191 ±0.025 | 0.565 ±0.016 | 0.758 ±0.035 | 0.956 ±0.007 | 1.14 ±0.04 | n/a | n/a |
| 0012 | 150 | 16.29 | 2.01 | 3.407 ±0.064 | 3.168 ±0.027 | 0.614 ±0.015 | 0.769 ±0.035 | 0.969 ±0.005 | 1.09 ±0.04 | n/a | n/a |
| 0013 | 150 | 15.02 | 2.83 | 3.395 ±0.058 | 3.223 ±0.029 | 0.640 ±0.015 | 0.739 ±0.038 | 0.980 ±0.003 | 1.03 ±0.03 | n/a | n/a |
| 0014 | 150 | 13.90 | 2.10 | 3.545 ±0.059 | 3.193 ±0.030 | 0.618 ±0.016 | 0.812 ±0.036 | 0.972 ±0.004 | 1.03 ±0.03 | n/a | n/a |
| 0015 | 150 | 14.03 | 1.83 | 3.673 ±0.055 | 3.161 ±0.034 | 0.596 ±0.016 | 0.801 ±0.036 | 0.980 ±0.003 | 1.04 ±0.04 | n/a | n/a |
| 0016 | 150 | 14.04 | 2.47 | 3.488 ±0.065 | 3.149 ±0.039 | 0.571 ±0.017 | 0.760 ±0.042 | 0.975 ±0.004 | 1.06 ±0.03 | n/a | n/a |
| 0017 | 150 | 15.38 | 2.73 | 3.644 ±0.058 | 3.148 ±0.034 | 0.527 ±0.017 | 0.722 ±0.049 | 0.981 ±0.002 | 1.04 ±0.04 | n/a | n/a |
| 0018 | 150 | 14.73 | 1.27 | 3.751 ±0.063 | 3.211 ±0.033 | 0.570 ±0.015 | 0.790 ±0.031 | 0.971 ±0.004 | 1.03 ±0.03 | n/a | n/a |
| 0019 | 150 | 15.62 | 2.74 | 3.673 ±0.058 | 3.237 ±0.029 | 0.482 ±0.017 | 0.755 ±0.041 | 0.970 ±0.004 | 1.10 ±0.05 | n/a | n/a |
| 0020 | 150 | 13.05 | 1.91 | 3.278 ±0.059 | 3.245 ±0.029 | 0.617 ±0.015 | 0.736 ±0.038 | 0.955 ±0.007 | 1.12 ±0.04 | n/a | n/a |

cross_label/esd by_emotion:
| group | n | WER% raw | WER% whisper-norm | UTMOSv2 | DNSMOS ovr | Spk cos | Emo cos | Acc cos | pred/GT dur | pred->target label | GT->target label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Angry | 300 | 13.30 | 3.10 | 3.500 ±0.044 | 3.127 ±0.024 | 0.493 ±0.012 | 0.803 ±0.017 | n/a | 1.03 ±0.02 | 0.61 | 0.98 |
| Happy | 300 | 11.56 | 2.05 | 3.514 ±0.043 | 3.123 ±0.024 | 0.446 ±0.015 | 0.582 ±0.018 | n/a | 1.10 ±0.02 | 0.09 | 0.93 |
| Neutral | 300 | 13.64 | 2.69 | 3.475 ±0.043 | 3.049 ±0.027 | 0.555 ±0.012 | 0.854 ±0.011 | n/a | 0.84 ±0.02 | 0.92 | 1.00 |
| Sad | 300 | 13.50 | 2.74 | 3.453 ±0.040 | 3.177 ±0.022 | 0.449 ±0.013 | 0.547 ±0.016 | n/a | 1.07 ±0.03 | 0.07 | 0.99 |
| Surprise | 300 | 12.84 | 2.46 | 3.528 ±0.043 | 3.113 ±0.026 | 0.439 ±0.011 | 0.292 ±0.026 | n/a | 1.06 ±0.03 | 0.07 | 0.95 |

cross_label/esd by_speaker:
| group | n | WER% raw | WER% whisper-norm | UTMOSv2 | DNSMOS ovr | Spk cos | Emo cos | Acc cos | pred/GT dur | pred->target label | GT->target label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0011 | 150 | 12.91 | 3.28 | 3.438 ±0.056 | 3.116 ±0.030 | 0.467 ±0.015 | 0.634 ±0.039 | n/a | 1.10 ±0.04 | n/a | n/a |
| 0012 | 150 | 13.71 | 3.11 | 3.199 ±0.054 | 2.953 ±0.041 | 0.495 ±0.019 | 0.598 ±0.041 | n/a | 1.03 ±0.03 | n/a | n/a |
| 0013 | 150 | 13.31 | 3.01 | 3.423 ±0.056 | 3.207 ±0.026 | 0.554 ±0.014 | 0.608 ±0.041 | n/a | 0.91 ±0.03 | n/a | n/a |
| 0014 | 150 | 14.57 | 2.83 | 3.373 ±0.060 | 3.027 ±0.037 | 0.522 ±0.017 | 0.609 ±0.044 | n/a | 0.94 ±0.03 | n/a | n/a |
| 0015 | 150 | 12.89 | 2.74 | 3.684 ±0.048 | 3.138 ±0.031 | 0.533 ±0.014 | 0.615 ±0.044 | n/a | 1.04 ±0.04 | n/a | n/a |
| 0016 | 150 | 12.71 | 2.47 | 3.525 ±0.048 | 3.087 ±0.035 | 0.472 ±0.018 | 0.608 ±0.043 | n/a | 1.13 ±0.04 | n/a | n/a |
| 0017 | 150 | 12.74 | 2.45 | 3.629 ±0.051 | 3.150 ±0.034 | 0.375 ±0.019 | 0.547 ±0.041 | n/a | 1.04 ±0.04 | n/a | n/a |
| 0018 | 150 | 12.46 | 1.91 | 3.805 ±0.053 | 3.158 ±0.033 | 0.461 ±0.018 | 0.692 ±0.035 | n/a | 1.05 ±0.04 | n/a | n/a |
| 0019 | 150 | 12.86 | 2.28 | 3.561 ±0.051 | 3.126 ±0.037 | 0.369 ±0.017 | 0.566 ±0.042 | n/a | 1.01 ±0.03 | n/a | n/a |
| 0020 | 150 | 11.55 | 2.00 | 3.301 ±0.057 | 3.215 ±0.025 | 0.514 ±0.015 | 0.679 ±0.037 | n/a | 0.93 ±0.03 | n/a | n/a |
