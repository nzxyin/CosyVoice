"""Synthesize the articulatory-tts held-out test splits with Fun-CosyVoice3-0.5B-2512
(zero-shot voice cloning), one 24 kHz wav per test utterance plus a JSONL manifest that
eval/score_synthesized.py consumes.

Test sets mirror /home/xoy/articulatory-tts/eval_full_testset.py's DATASET_SPECS exactly (same
split files, same raw-audio/transcript sources), so the utterance lists are identical to that
repo's eval_*.json files and to the XTTS (/home/xoy/TTS/eval) and EmoSphere++
(/home/xoy/EmoSpherepp/eval) baseline evaluations:

  ljspeech             LJSpeech-1.1/preprocessed/test.json            (150 utts, 1 speaker)
  libritts_test_clean  LibriTTS_R/test-clean.json                     (4830 utts, 39 speakers)
  libritts_test_other  LibriTTS_R/test-other.json                     (5106 utts, 33 speakers)
  esd                  esd_english_splits/test.tsv                    (1500 utts, 10 speakers x 5 emotions)
  vctk                 vctk_globe_accent_splits/vctk_only/test.tsv    (2596 utts, 6 held-out speakers, one per accent)

Inference = CosyVoice3's standard zero-shot API, AutoModel.inference_zero_shot(text, prompt_text,
prompt_wav): the prompt recording's speech tokens + transcript are the in-context prefix for the
LLM, its mel + CAM++ x-vector condition the flow-matching decoder. prompt_text is prefixed with
"You are a helpful assistant.<|endofprompt|>" exactly as example.py / the libritts CosyVoice3
recipe do (every CosyVoice3 training sequence carries that instruct). Default sampling from the
checkpoint's cosyvoice3.yaml (RAS sampling, top_p 0.8, top_k 25), fp32, non-streaming, speed 1.0,
text_frontend=True (wetext English normalization + inflect number spelling, the shipped default;
--no_text_frontend disables it). Long texts are split by CosyVoice's own paragraph splitter
(<=80 text tokens per segment, same prompt for every segment) and the segments concatenated.

Speaker prompt (--prompt_mode):
  self   (default) the target utterance's own ground-truth recording (+ its own transcript as
         prompt_text). This is the protocol the articulatory-tts numbers use (its SPARC vocoder is
         driven by the target utterance's own speaker embedding) and what the XTTS/EmoSphere++
         baselines ran, so use it for numbers meant to sit next to those. CAVEAT specific to an
         in-context LLM like CosyVoice: in this mode the model is shown the speech tokens of the
         very sentence it must produce, i.e. it is asked to "say the same sentence again" -- an
         optimistic condition for WER/speaker/emotion/accent similarity relative to true zero-shot.
  cross  a different utterance from the same speaker within the same test split (deterministic:
         the next uid in that speaker's sorted list, cyclic; for ESD the group is speaker x emotion
         so the prompt carries the target emotion). The standard zero-shot TTS protocol (Seed-TTS
         eval / CV3-Eval style); report alongside `self`. On ESD this is the EMOTION-CLONING
         condition: the emotion reaches the model only through the prompt recording. The pairing is
         identical, stem for stem, to articulatory-tts's emotion_ref_mode=cross (GH #91,
         esd_english_splits/test_cross_ref_pairs.tsv; verified on all 1500 test rows 2026-09-24).
  cross_label  (ESD only) the EMOTION-LABEL condition: the target emotion is given as a text
         instruction and the prompt recording is a different, NEUTRAL utterance of the same speaker
         (the i-th target of a (speaker, emotion) group gets the (i+1)-th, cyclic, of that speaker's
         Neutral group -- for Neutral targets exactly the `cross` pairing). Inference goes through
         CosyVoice3's instruct API, inference_instruct2(text, instruct, prompt_wav): the instruct
         (--emotion_instruct_set, see EMOTION_INSTRUCT_SETS) replaces the prompt transcript, and the
         prompt's speech tokens are NOT shown to the LLM (frontend_instruct2 drops them), so the
         recording reaches only the flow decoder (mel prompt + CAM++ x-vector), i.e. the voice. This
         is the counterpart of articulatory-tts's categorical-label conditioning (and EmoSphere++'s
         emotion-ID input) to the reference-derived cloning condition above.

Prompt length: CosyVoice's speech tokenizer rejects prompts > 30 s, so a longer prompt (a few
LibriTTS-R utterances) is cropped to its first PROMPT_CROP_SEC seconds (written once under
<out_dir>/prompt_crops/); the manifest records prompt_seconds_used.

Restart-safe: an existing non-empty output wav is skipped, so a preempted job resumes where it
stopped. Sharding (--shard_index/--num_shards) splits the utterance list round-robin after a
deterministic sort. The seed is fixed per utterance (seed + crc32(uid)) so a rerun reproduces the
same sample.

Usage (see eval/run_synth.sbatch):
  python eval/synthesize_testset.py --dataset esd --prompt_mode self \
      --model_dir /data/user_data/xoy/cosyvoice_models/Fun-CosyVoice3-0.5B-2512 \
      --out_dir /data/user_data/xoy/cosyvoice3_eval/Fun-CosyVoice3-0.5B-2512/self/esd
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback
import zlib

# numpy/torch/soundfile are imported inside functions: this module's DATASETS/load_items/
# assign_prompts are also used by eval/run_score.sbatch's completeness gate (system python3, no
# numpy) and by eval/score_synthesized.py (a different venv), so the top level must stay stdlib-only.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- test-set definitions (paths identical to articulatory-tts's eval_full_testset.py) ----------
DATASETS = {
    "ljspeech": {
        "split_path": "/data/user_data/xoy/LJSpeech-1.1/preprocessed/test.json",
        "raw_wav_dir": "/data/user_data/xoy/LJSpeech-1.1/wavs",
        "metadata_csv": "/data/user_data/xoy/LJSpeech-1.1/metadata.csv",
    },
    "libritts_test_clean": {
        "split_path": "/data/user_data/xoy/LibriTTS_R/test-clean.json",
        "raw_wav_dir": "/data/user_data/xoy/LibriTTS_R/test-clean",
    },
    "libritts_test_other": {
        "split_path": "/data/user_data/xoy/LibriTTS_R/test-other.json",
        # no local copy of test-other exists; same group_data source the reference eval reads
        "raw_wav_dir": "/data/group_data/UTD-NAS/Databases/LibriTTS-R/LibriTTS_R/test-other",
    },
    "esd": {
        # ESD's own official test partition (all 10 English speakers x 5 emotions x 30 utts);
        # columns: stem, source, speaker (esd_0011), emotion. Transcripts come from the per-speaker
        # <spk>/<spk>.txt tables under raw_wav_dir (the "ESD/ESD" subtree specifically, not its
        # "Emotion Speech Dataset"/__MACOSX duplicate siblings), same as the reference.
        "split_path": "/data/user_data/xoy/esd_english_splits/test.tsv",
        "raw_wav_dir": "/data/group_data/UTD-NAS/Databases/ESD/ESD",
    },
    # ESD's official val partition (1000 utts, 10 speakers x 5 emotions x 20), disjoint from test.
    # Only used to choose the --emotion_instruct_set wording (eval/pilot_emotion_instruct.sbatch), so
    # that choice is not tuned on the test set.
    "esd_val": {
        "split_path": "/data/user_data/xoy/esd_english_splits/val.tsv",
        "raw_wav_dir": "/data/group_data/UTD-NAS/Databases/ESD/ESD",
    },
    "vctk": {
        "split_path": "/data/user_data/xoy/vctk_globe_accent_splits/vctk_only/test.tsv",
        "raw_wav_dir": "/data/group_data/UTD-NAS/Databases/VCTK/VCTK-Corpus/wav48",
        "txt_dir": "/data/group_data/UTD-NAS/Databases/VCTK/VCTK-Corpus/txt",
        # TsvCorpusDataset drops split rows with no phn_ids_2 entry (upstream phonemization gap) --
        # mirror that so the utterance set is identical to what eval_full_testset.py scores.
        # Currently a no-op (2596/2596 present), kept so the two stay in lockstep.
        "phn_ids_dir": "/data/user_data/xoy/VCTK/VCTK-Corpus/preprocessed/phn_ids_2",
    },
    # The two speakers articulatory-tts added to its VCTK test set on 2026-09-16:
    # SouthAfrican p336 and Indian p251, 782 utterances,
    # vctk_only_plus_sa_in/test_new_accents.tsv. A separate set, so the
    # 2,596-utterance `vctk` results above stay unchanged.
    "vctk_new_accents": {
        "split_path": "/data/user_data/xoy/vctk_globe_accent_splits/vctk_only_plus_sa_in/test_new_accents.tsv",
        "raw_wav_dir": "/data/group_data/UTD-NAS/Databases/VCTK/VCTK-Corpus/wav48",
        "txt_dir": "/data/group_data/UTD-NAS/Databases/VCTK/VCTK-Corpus/txt",
        "phn_ids_dir": "/data/user_data/xoy/VCTK/VCTK-Corpus/preprocessed/phn_ids_2",
    },
}

# CosyVoice3 instruct prefix every prompt_text carries (example.py, examples/libritts/cosyvoice3).
INSTRUCT_PREFIX = "You are a helpful assistant.<|endofprompt|>"

PROMPT_MODES = ["self", "cross", "cross_label"]

# --prompt_mode cross_label: ESD emotion -> instruction, sent as
# "You are a helpful assistant. <instruction><|endofprompt|>". "zh" follows the only emotion instructs
# the repo lists for CosyVoice3 (cosyvoice/utils/common.py instruct_list: 开心 / 伤心 / 生气 in the
# "请非常X地说一句话。" template), extended with the same template to Surprise and a calm-tone phrase
# for Neutral. "en" is the English form of the same template, modeled on the list's English volume
# instructs ("Please say a sentence as loudly as possible."). "none" gives no emotion instruction at
# all (control: Neutral prompt through the instruct API only). The default, "en_plain_neutral" = "en"
# for the four emotions + no instruction for Neutral, was chosen per emotion on ESD val (pilot job
# 10552276, 20 utts per emotion, paired bootstrap): en beat zh on each non-Neutral emotion (+0.061
# emotion cosine [+0.033, +0.089] pooled), but en's Neutral phrase was worse than no instruction
# (-0.069 [-0.120, -0.020]; zh's Neutral phrase vs none was a tie, +0.023 [-0.005, +0.051]).
# See eval/pilot_emotion_instruct.sbatch and CLAUDE.md.
EMOTION_INSTRUCT_SETS = {
    "zh": {"Angry": "请非常生气地说一句话。", "Happy": "请非常开心地说一句话。", "Sad": "请非常伤心地说一句话。",
           "Surprise": "请非常惊讶地说一句话。", "Neutral": "请用平静的语气说一句话。"},
    "en": {"Angry": "Please say a sentence in a very angry tone.",
           "Happy": "Please say a sentence in a very happy tone.",
           "Sad": "Please say a sentence in a very sad tone.",
           "Surprise": "Please say a sentence in a very surprised tone.",
           "Neutral": "Please say a sentence in a calm, neutral tone."},
    "none": {},
}
EMOTION_INSTRUCT_SETS["en_plain_neutral"] = {e: p for e, p in EMOTION_INSTRUCT_SETS["en"].items() if e != "Neutral"}
DEFAULT_EMOTION_INSTRUCT_SET = "en_plain_neutral"


def emotion_instruct_text(emotion, instruct_set):
    """The instruct_text inference_instruct2 gets for one target emotion."""
    phrase = EMOTION_INSTRUCT_SETS[instruct_set].get(emotion)
    if not phrase and not (instruct_set == "none" or (instruct_set == "en_plain_neutral" and emotion == "Neutral")):
        raise ValueError(f"no {instruct_set!r} instruction for emotion {emotion!r}")
    return f"You are a helpful assistant. {phrase}<|endofprompt|>" if phrase else INSTRUCT_PREFIX
# cosyvoice/cli/frontend.py asserts prompt <= 30 s; crop with a margin.
MAX_PROMPT_SEC = 30.0
PROMPT_CROP_SEC = 29.0

SPLIT_STATS = {"n_split_total": None}  # filled by load_items()


def read_text(path):
    """VCTK/LibriTTS transcripts are overwhelmingly utf-8; same utf-8 -> latin-1 fallback as
    eval_full_testset.py uses for VCTK."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(path, encoding="latin-1") as f:
            return f.read().strip()


def read_esd_speaker_txt(raw_wav_dir, spk):
    """ESD's per-speaker <spk>/<spk>.txt (tab-separated stem / transcript / emotion). Mixed
    encodings across speakers (utf-8, BOM-prefixed utf-16, one single-byte file) -- BOM sniff
    first, then utf-8-sig, then latin-1 (same chain the reference/EmoSphere++ evals settled on)."""
    with open(os.path.join(raw_wav_dir, spk, f"{spk}.txt"), "rb") as f:
        raw = f.read()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeError:
            text = raw.decode("latin-1")
    rows = {}
    for line in text.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            rows[parts[0]] = (parts[1].strip(), parts[2].strip())
        elif len(parts) == 2:
            rows[parts[0]] = (parts[1].strip(), None)
    return rows


def index_wavs(raw_wav_dir):
    """stem -> path, one os.walk (same trick as eval_full_testset.py's wav_index)."""
    idx = {}
    for root, _dirs, files in os.walk(raw_wav_dir):
        for fn in files:
            if fn.endswith(".wav"):
                idx.setdefault(fn[:-4], os.path.join(root, fn))
    return idx


def load_items(dataset):
    spec = DATASETS[dataset]
    items = []
    if dataset == "ljspeech":
        ids = json.load(open(spec["split_path"]))
        id_set = set(ids)
        texts = {}
        with open(spec["metadata_csv"], encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE):
                if row and row[0] in id_set:
                    # column 3 = normalized transcript (numbers spelled out), same column the reference WER uses
                    texts[row[0]] = row[2] if len(row) > 2 else row[1]
        for uid in ids:
            items.append({
                "uid": uid, "speaker": "LJ", "text": texts.get(uid),
                "gt_wav": os.path.join(spec["raw_wav_dir"], f"{uid}.wav"),
                "emotion": "unknown", "accent_label": "unknown",
            })
    elif dataset.startswith("libritts"):
        ids = json.load(open(spec["split_path"]))
        for uid in ids:
            spk, chap = uid.split("_")[0], uid.split("_")[1]
            d = os.path.join(spec["raw_wav_dir"], spk, chap)
            txt = os.path.join(d, f"{uid}.normalized.txt")
            items.append({
                "uid": uid, "speaker": spk,
                "text": read_text(txt) if os.path.exists(txt) else None,
                "gt_wav": os.path.join(d, f"{uid}.wav"),
                "emotion": "unknown", "accent_label": "unknown",
            })
    elif dataset.startswith("esd"):
        with open(spec["split_path"], newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        wav_index = index_wavs(spec["raw_wav_dir"])
        tables = {}
        for r in rows:
            spk = r["stem"].split("_")[0]
            if spk not in tables:
                tables[spk] = read_esd_speaker_txt(spec["raw_wav_dir"], spk)
            txt, emo_txt = tables[spk].get(r["stem"], (None, None))
            emotion = r.get("emotion") or emo_txt or "unknown"
            if emo_txt and r.get("emotion") and emo_txt != r["emotion"]:
                print(f"WARNING: {r['stem']} emotion mismatch split={r['emotion']} txt={emo_txt}; using split")
            items.append({
                "uid": r["stem"], "speaker": spk, "text": txt,
                "gt_wav": wav_index.get(r["stem"], os.path.join(spec["raw_wav_dir"], spk, emotion, f"{r['stem']}.wav")),
                "emotion": emotion, "accent_label": "unknown",
            })
    elif dataset.startswith("vctk"):
        with open(spec["split_path"], newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        available = {fn[: -len(".phn.npy")] for fn in os.listdir(spec["phn_ids_dir"]) if fn.endswith(".phn.npy")}
        dropped = [r["stem"] for r in rows if r["stem"] not in available]
        if dropped:
            print(f"WARNING: {len(dropped)}/{len(rows)} VCTK test rows have no phn_ids_2 entry and are "
                  f"excluded (same filter as articulatory-tts's TsvCorpusDataset), e.g. {dropped[:3]}")
        for r in rows:
            if r["stem"] in dropped:
                continue
            spk = r["speaker"]
            txt = os.path.join(spec["txt_dir"], spk, f"{r['stem']}.txt")
            items.append({
                "uid": r["stem"], "speaker": spk,
                "text": read_text(txt) if os.path.exists(txt) else None,
                "gt_wav": os.path.join(spec["raw_wav_dir"], spk, f"{r['stem']}.wav"),
                "emotion": "unknown", "accent_label": r["accent"],
            })
    else:
        raise ValueError(dataset)

    n_all = len(items)
    # CosyVoice needs the transcript (it is the text to speak) and the GT recording (it is the
    # prompt in self mode and the scorer's reference in both modes), so rows lacking either cannot
    # be evaluated. A no-op on these five splits (0 rows lack either), reported if it ever isn't.
    items = [it for it in items if it["text"] and os.path.exists(it["gt_wav"])]
    SPLIT_STATS["n_split_total"] = n_all
    if len(items) != n_all:
        print(f"WARNING: {n_all - len(items)}/{n_all} utterances dropped for missing transcript or ground-truth wav")
    items.sort(key=lambda it: it["uid"])  # deterministic order before sharding / cross-prompt pairing
    return items


def prompt_group_key(dataset, it):
    """Utterances eligible to prompt each other in --prompt_mode cross: same speaker, and for ESD
    also the same emotion (the target emotion is not inferable from the text alone)."""
    return (it["speaker"], it["emotion"]) if dataset.startswith("esd") else (it["speaker"],)


def assign_prompts(items, mode, dataset):
    """Sets prompt_wav / prompt_text / prompt_uid on every item (items uid-sorted, as load_items
    returns them)."""
    if mode == "self":
        for it in items:
            it["prompt_uid"], it["prompt_wav"], it["prompt_text"] = it["uid"], it["gt_wav"], it["text"]
        return
    if mode == "cross_label" and not dataset.startswith("esd"):
        raise ValueError("--prompt_mode cross_label needs emotion labels (ESD only)")
    groups = {}
    for it in items:
        groups.setdefault(prompt_group_key(dataset, it), []).append(it)
    for key, group in groups.items():
        if mode == "cross_label":
            # i-th member of (speaker, emotion) -> (i+1)-th of (speaker, Neutral), cyclic. Never the target
            # itself: a non-Neutral target is not in the Neutral group, and a one-member Neutral group raises.
            source = groups.get((key[0], "Neutral"))
            if not source:
                raise ValueError(f"speaker {key[0]} has no Neutral utterance to prompt {key} with")
            if key[1] == "Neutral" and len(source) == 1:
                raise ValueError(f"Neutral group {key} has a single utterance; no cross prompt possible")
        else:
            source = group
            if len(group) == 1:
                print(f"WARNING: group {key} has a single test utterance; cross prompt falls back to self")
        for i, it in enumerate(group):
            p = source[(i + 1) % len(source)]
            it["prompt_uid"], it["prompt_wav"], it["prompt_text"] = p["uid"], p["gt_wav"], p["text"]


# results-JSON name of each prompt mode's emotion condition on ESD (articulatory-tts records its
# counterpart as "emotion_ref_mode": self | cross, next to the stem-by-stem "emotion_refs")
EMOTION_REF_MODE = {"self": "self", "cross": "cross", "cross_label": "label"}


def emotion_ref_provenance(prompt_mode, items, split_info=None):
    """ESD provenance for a results JSON: which emotion condition, and the prompt recording each
    target was conditioned on (items after assign_prompts)."""
    out = {"emotion_ref_mode": EMOTION_REF_MODE[prompt_mode],
           "emotion_refs": {it["uid"]: it["prompt_uid"] for it in items}}
    if prompt_mode == "cross_label":
        out["emotion_instruct_set"] = (split_info or {}).get("emotion_instruct_set")
        out["emotion_instructs"] = (split_info or {}).get("emotion_instructs")
    return out


def prepare_prompt(prompt_wav, crop_dir):
    """Return (path, seconds_used): the prompt itself, or a <= PROMPT_CROP_SEC crop written once
    under crop_dir when the recording exceeds the tokenizer's 30 s limit."""
    import soundfile as sf
    with sf.SoundFile(prompt_wav) as f:
        sr, n = f.samplerate, len(f)
    dur = n / sr
    if dur <= MAX_PROMPT_SEC - 0.05:
        return prompt_wav, dur
    os.makedirs(crop_dir, exist_ok=True)
    out = os.path.join(crop_dir, os.path.basename(prompt_wav))
    if not (os.path.exists(out) and os.path.getsize(out) > 1000):
        wav, sr = sf.read(prompt_wav, frames=int(PROMPT_CROP_SEC * sr), dtype="float32")
        tmp = out + ".tmp.wav"
        sf.write(tmp, wav, sr)
        os.replace(tmp, out)
    return out, PROMPT_CROP_SEC


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--model_dir", required=True, help="local Fun-CosyVoice3-0.5B-2512 dir (cosyvoice3.yaml, llm.pt, flow.pt, hift.pt, ...)")
    ap.add_argument("--out_dir", required=True, help="wavs go to <out_dir>/wavs/<uid>.wav; manifest JSONL alongside")
    ap.add_argument("--prompt_mode", choices=PROMPT_MODES, default="self")
    ap.add_argument("--emotion_instruct_set", choices=list(EMOTION_INSTRUCT_SETS), default=DEFAULT_EMOTION_INSTRUCT_SET,
                    help="--prompt_mode cross_label only: wording of the emotion instruction (EMOTION_INSTRUCT_SETS)")
    ap.add_argument("--no_text_frontend", action="store_true",
                    help="pass text_frontend=False: skips wetext normalization / number spelling of the target text AND "
                         "CosyVoice's <=80-token paragraph splitting (frontend.text_normalize returns the text whole)")
    ap.add_argument("--fp16", action="store_true", help="AutoModel(fp16=True); default fp32")
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="only the first N utterances of this shard (smoke test)")
    ap.add_argument("--seed", type=int, default=0, help="per-utterance seed = seed + crc32(uid), for reproducible sampling")
    ap.add_argument("--max_attempts", type=int, default=3,
                    help="re-sample (seed + attempt*1000003) when synthesis raises the short-output vocoder error "
                         "(LLM emitted ~no speech tokens for a very short text) or yields no audio; attempts are recorded")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    items = load_items(args.dataset)
    assign_prompts(items, args.prompt_mode, args.dataset)
    label_mode = args.prompt_mode == "cross_label"
    if label_mode:
        for it in items:
            it["instruct_text"] = emotion_instruct_text(it["emotion"], args.emotion_instruct_set)
    shard = items[args.shard_index::args.num_shards]
    if args.limit:
        shard = shard[: args.limit]
    print(f"{args.dataset}: {len(items)} utterances total, {len(shard)} in shard "
          f"{args.shard_index}/{args.num_shards}, prompt_mode={args.prompt_mode}")

    wav_dir = os.path.join(args.out_dir, "wavs")
    crop_dir = os.path.join(args.out_dir, "prompt_crops")
    os.makedirs(wav_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, f"manifest.shard{args.shard_index}of{args.num_shards}.jsonl")
    info_path = os.path.join(args.out_dir, "split_info.json")
    if not os.path.exists(info_path):
        # Shards of the same set start within seconds of each other and all race to write this file
        # (job 10354158 died on a sibling's os.replace of the shared .tmp name): per-shard temp name,
        # and losing the race is fine -- the content is identical.
        tmp = f"{info_path}.shard{args.shard_index}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"dataset": args.dataset, "split_path": DATASETS[args.dataset]["split_path"],
                       "n_split_total": SPLIT_STATS["n_split_total"], "n_synthesizable": len(items),
                       "n_dropped_missing_text_or_gt_wav": SPLIT_STATS["n_split_total"] - len(items),
                       "prompt_mode": args.prompt_mode, "model_dir": args.model_dir,
                       "text_frontend": not args.no_text_frontend, "fp16": args.fp16,
                       "instruct_prefix": INSTRUCT_PREFIX,
                       **({"emotion_instruct_set": args.emotion_instruct_set,
                           "emotion_instructs": {e: emotion_instruct_text(e, args.emotion_instruct_set)
                                                 for e in sorted({it["emotion"] for it in items})}}
                          if label_mode else {})}, f, indent=2)
        try:
            os.replace(tmp, info_path)
        except FileNotFoundError:
            pass  # a sibling shard won the race

    # everything below needs the model; import late so --help / the completeness gate stay cheap
    import logging
    import random
    import numpy as np
    import soundfile as sf
    import torch
    sys.path.insert(0, REPO_ROOT)
    sys.path.insert(0, os.path.join(REPO_ROOT, "third_party", "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible -- refusing to run CosyVoice3 on CPU")
    t0 = time.time()
    model = AutoModel(model_dir=args.model_dir, fp16=args.fp16)
    out_sr = model.sample_rate
    print(f"loaded {args.model_dir} ({type(model).__name__}, fp16={args.fp16}, sample_rate={out_sr}) in "
          f"{time.time() - t0:.0f}s on {torch.cuda.get_device_name(0)}; text_frontend={not args.no_text_frontend}")
    # cosyvoice.utils.file_utils sets the root logger to DEBUG at import; CosyVoice then logs every
    # segment and rtf. Quiet that down -- this script prints its own progress lines.
    logging.getLogger().setLevel(logging.WARNING)

    n_done = n_skipped = n_failed = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 10
    t_start = time.time()
    gen_time = 0.0
    audio_sec = 0.0
    with open(manifest_path, "a") as mf:
        for k, it in enumerate(shard):
            pred_path = os.path.join(wav_dir, f"{it['uid']}.wav")
            record = {**it, "pred_wav": pred_path, "prompt_mode": args.prompt_mode,
                      "text_frontend": not args.no_text_frontend, "fp16": args.fp16}
            if not args.overwrite and os.path.exists(pred_path) and os.path.getsize(pred_path) > 1000:
                n_skipped += 1
                record["status"] = "existing"
                mf.write(json.dumps(record) + "\n")
                continue
            base_seed = (args.seed + zlib.crc32(it["uid"].encode())) % (2**31)
            seed = base_seed
            t1 = time.time()
            try:
                prompt_path, prompt_sec = prepare_prompt(it["prompt_wav"], crop_dir)
                record["prompt_seconds_used"] = round(prompt_sec, 3)
                prompt_text = INSTRUCT_PREFIX + it["prompt_text"]
                # What the model will actually be asked to say: the frontend's normalized + split segments
                # (deterministic; inference_zero_shot recomputes the same thing). Recorded so WER can also be
                # scored against the spoken text, not only the raw reference transcript.
                record["normalized_text"] = [str(seg) for seg in model.frontend.text_normalize(
                    it["text"], split=True, text_frontend=not args.no_text_frontend)]
                wav = None
                for attempt in range(max(1, args.max_attempts)):
                    seed = (base_seed + attempt * 1000003) % (2**31)
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    try:
                        chunks = []
                        with torch.no_grad():
                            if label_mode:
                                # the instruct replaces the prompt transcript; the prompt's speech tokens are
                                # dropped for the LLM (frontend_instruct2) and kept for the flow decoder
                                gen = model.inference_instruct2(it["text"], it["instruct_text"], prompt_path, stream=False,
                                                                speed=1.0, text_frontend=not args.no_text_frontend)
                            else:
                                gen = model.inference_zero_shot(it["text"], prompt_text, prompt_path, stream=False, speed=1.0,
                                                                text_frontend=not args.no_text_frontend)
                            for out in gen:
                                chunks.append(out["tts_speech"])
                        if not chunks:
                            raise RuntimeError("model yielded no audio")
                        wav = torch.cat(chunks, dim=1).squeeze(0).float().cpu().numpy()
                        if wav.size == 0:
                            raise RuntimeError("empty waveform")
                        break
                    except RuntimeError as e:
                        # The LLM occasionally emits (almost) no speech tokens for a very short text ("no",
                        # "A watch."); HiFT's f0 predictor then fails with "Kernel size can't be greater than
                        # actual input size" on a 2-3 frame mel. Sampling is stochastic, so re-sample with a
                        # new seed a couple of times before giving up. CUDA errors are never retried.
                        retryable = ("Kernel size" in str(e) or "no audio" in str(e) or "empty waveform" in str(e)) \
                            and "CUDA" not in str(e)
                        if not retryable or attempt + 1 >= max(1, args.max_attempts):
                            raise
                        print(f"RETRY {it['uid']} attempt {attempt + 1} failed ({e}); re-sampling with a new seed",
                              file=sys.stderr, flush=True)
                        wav = None
                record["attempts"] = attempt + 1
                tmp = pred_path + ".tmp.wav"
                sf.write(tmp, wav, out_sr)
                os.replace(tmp, pred_path)  # atomic: a preempted job never leaves a truncated final wav
                dt = time.time() - t1
                gen_time += dt
                audio_sec += len(wav) / out_sr
                n_done += 1
                record.update({"status": "ok", "seed": seed, "n_segments": len(chunks),
                               "pred_seconds": round(len(wav) / out_sr, 3), "gen_wall_seconds": round(dt, 2)})
            except Exception as e:
                n_failed += 1
                consecutive_failures += 1
                record.update({"status": "failed", "seed": seed, "error": f"{type(e).__name__}: {e}"})
                print(f"FAILED {it['uid']}: {type(e).__name__}: {e}", file=sys.stderr)
                traceback.print_exc()
                # A per-utterance failure is logged and skipped, but a systemic one must abort loudly:
                # a CUDA error (e.g. a GPU this torch build has no kernels for) or a run of
                # consecutive failures means nothing downstream will work either.
                if "CUDA" in str(e) or consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    mf.write(json.dumps(record) + "\n")
                    mf.flush()
                    print(f"ABORTING: {consecutive_failures} consecutive failures / CUDA error -- "
                          f"see traceback above", file=sys.stderr)
                    sys.exit(3)
            else:
                consecutive_failures = 0
            mf.write(json.dumps(record) + "\n")
            mf.flush()
            if (k + 1) % 25 == 0 or k + 1 == len(shard):
                elapsed = time.time() - t_start
                rate = gen_time / max(n_done, 1)
                remaining = sum(1 for it2 in shard[k + 1:]
                                if not os.path.exists(os.path.join(wav_dir, f"{it2['uid']}.wav"))) * rate
                rtf = gen_time / audio_sec if audio_sec else float("nan")
                print(f"[{k + 1}/{len(shard)}] done={n_done} skipped={n_skipped} failed={n_failed} "
                      f"elapsed={elapsed / 60:.1f}min avg_gen={rate:.2f}s/utt rtf={rtf:.2f} eta={remaining / 60:.1f}min", flush=True)

    print(f"SUMMARY dataset={args.dataset} shard={args.shard_index}/{args.num_shards} prompt_mode={args.prompt_mode} "
          f"synthesized={n_done} skipped_existing={n_skipped} failed={n_failed} "
          f"gen_time={gen_time / 60:.1f}min audio={audio_sec / 60:.1f}min manifest={manifest_path}")
    if n_failed and n_done == 0 and n_skipped == 0:
        sys.exit(1)
    # completion marker: only written when the loop ran to the end of the shard (a preempted job
    # leaves a manifest but no marker). run_score.sbatch's gate checks the wavs themselves.
    with open(manifest_path + ".done", "w") as f:
        json.dump({"shard": args.shard_index, "num_shards": args.num_shards, "n_shard": len(shard),
                   "synthesized": n_done, "skipped_existing": n_skipped, "failed": n_failed,
                   "limit": args.limit}, f)


if __name__ == "__main__":
    main()
