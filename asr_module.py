import os
import random
import torch
import torchaudio
import joblib
import pandas as pd
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from pyctcdecode import build_ctcdecoder
from model_definition import ASRModel

# === Configuration and Constants ===

SAMPLE_RATE = 16000
N_MELS = 128
CHARACTERS = ["'", " ", *list("abcdefghijklmnopqrstuvwxyz")]
BLANK = "_"
VOCAB = [BLANK] + CHARACTERS
char2idx = {c: i for i, c in enumerate(VOCAB)}
idx2char = {i: c for c, i in char2idx.items()}

# === CTC Beam Search Decoder Setup ===

vocab_list = VOCAB.copy()
vocab_list[0] = ""  # Remove blank token from decoder vocab
beam_decoder = build_ctcdecoder(vocab_list)

# === Audio Preprocessing Transforms ===

mel_transform = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=1024, hop_length=256)
db_transform = AmplitudeToDB()

# === Model Initialization ===

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASRModel(N_MELS, hidden_dim=512, vocab_size=len(VOCAB)).to(device)
model.load_state_dict(torch.load("best_asr_model.pth", map_location=device))
model.eval()

# === Load Pretrained Text Classifiers ===

action_classifier = joblib.load("action_classifier.pkl")
object_classifier = joblib.load("object_classifier.pkl")

# === Load and Prepare Test Labels ===

test_labels = pd.read_csv("test_labels.csv", header=None, names=["text", "action", "object"])
test_manifest = pd.read_csv("test.csv", header=None, names=["path", "transcript"]).dropna()
test_labels["file"] = test_manifest["path"].apply(lambda p: os.path.basename(str(p)))

action_map = {0: "off", 1: "on"}
file2label = dict(zip(
    test_labels["file"],
    zip(
        test_labels["text"],
        test_labels["action"].map(action_map),
        test_labels["object"]
    )
))

# === Helper Functions ===

def enrich_text(text):
    """Append 'music' keyword if music-related verbs are detected."""
    text = text.lower()
    music_keywords = ["stop", "pause", "resume", "continue", "play"]
    if any(kw in text for kw in music_keywords):
        text += " music"
    return text

def preprocess_audio(path):
    """Load and preprocess audio into mel-spectrogram format."""
    waveform, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
    spec = mel_transform(waveform)
    spec = db_transform(spec)
    return spec.squeeze(0).transpose(0, 1)  # [time, features]

def beam_search_decode(logits, input_lens):
    # Decode model output using CTC beam search.
    logits = logits.cpu().permute(1, 0, 2)  # [Batch, Time, Vocab]
    results = []
    for b in range(logits.shape[0]):
        log_probs = logits[b, :input_lens[b]]
        decoded = beam_decoder.decode(log_probs.numpy())
        results.append(decoded.strip())
    return results

def get_ground_truth(filename):
    """Return ground truth text/action/object for a given filename."""
    return file2label.get(filename, (None, None, None))

# === Inference Function ===

def infer_latest_command(test_dir="test_dataset"):
    """
    Run ASR + classification on a random audio file from the test set.
    Returns:
        - predicted text
        - predicted action label
        - predicted object label
        - tuple with ground truth (text, action, object)
    """
    files = [f for f in os.listdir(test_dir) if f.endswith(".wav")]
    if not files:
        return "", "", "", None

    selected = random.choice(files)
    path = os.path.join(test_dir, selected)

    # Inference: audio -> spectrogram -> model -> decoded text
    spec = preprocess_audio(path)
    input_tensor = spec.unsqueeze(1).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        input_len = torch.tensor([logits.size(0)])
        text = beam_search_decode(logits, input_len)[0]

    # Classification based on decoded text
    action = action_classifier.predict([text])[0]
    obj = object_classifier.predict([enrich_text(text)])[0]

    # Ground truth for comparison
    true_text, true_action, true_object = get_ground_truth(selected)

    return text, action, obj, (true_text, true_action, true_object)
