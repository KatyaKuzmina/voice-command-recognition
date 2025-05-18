import os
import pandas as pd
import whisper
from tqdm import tqdm
from sklearn.metrics import classification_report
from torchaudio.functional import edit_distance
import joblib

# === Test Data Preparation ===

test_df = pd.read_csv("test_labels.csv", header=None, names=["text", "action", "object"])
test_texts = test_df["text"].tolist()
test_actions = test_df["action"].map({0: "off", 1: "on"}).tolist()
test_objects = test_df["object"].tolist()

with open("test.csv", "r", encoding="utf-8") as f:
    audio_manifest = [line.strip().split(",")[0] for line in f.readlines()]
audio_manifest = [os.path.join("test_dataset", os.path.basename(p)) for p in audio_manifest]


# === Text Enrichment Function ===

def enrich_text(text):
    text = text.lower()
    music_keywords = ["stop", "pause", "resume", "continue", "play"]
    if any(kw in text for kw in music_keywords):
        text += " music"
    return text


# === WER / CER Calculation ===

def compute_wer_cer(refs, hyps):
    total_wer, total_cer = 0, 0
    for r, h in zip(refs, hyps):
        total_wer += edit_distance(r.split(), h.split()) / max(1, len(r.split()))
        total_cer += edit_distance(list(r), list(h)) / max(1, len(r))
    return total_wer / len(refs), total_cer / len(refs)


# === Load Classifiers ===

action_classifier = joblib.load("action_classifier.pkl")
object_classifier = joblib.load("object_classifier.pkl")

# === Run Evaluation for Multiple Whisper Models (Clean Audio) ===

for model_size in ["small", "medium", "large"]:
    print(f"\nTranscribing with Whisper-{model_size} on clean test data...")
    model = whisper.load_model(model_size)

    whisper_texts = []
    for path in tqdm(audio_manifest, desc=f"Whisper-{model_size} Transcribing"):
        result = model.transcribe(path)
        whisper_texts.append(result["text"].lower().strip())

    whisper_actions = [action_classifier.predict([t])[0] for t in whisper_texts]
    whisper_objects = [object_classifier.predict([enrich_text(t)])[0] for t in whisper_texts]

    print(f"\nWhisper-{model_size.upper()} — Action Classification (on/off):")
    print(classification_report(test_actions[:len(whisper_actions)], whisper_actions, labels=["off", "on"]))

    print(f"Whisper-{model_size.upper()} — Object Classification (lights/music):")
    print(classification_report(test_objects[:len(whisper_objects)], whisper_objects, labels=["lights", "music"]))

    wer, cer = compute_wer_cer(test_texts[:len(whisper_texts)], whisper_texts)
    print(f"Whisper-{model_size.upper()} — WER: {wer:.3f}, CER: {cer:.3f}")
