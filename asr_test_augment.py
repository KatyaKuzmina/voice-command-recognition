import os
import csv
import time
import torch
import torchaudio
import pandas as pd
import random
import joblib
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchaudio.functional import edit_distance
from torch.utils.data import Dataset, DataLoader
from pyctcdecode import build_ctcdecoder

# === Settings and Constants ===

SAMPLE_RATE = 16000
N_MELS = 128
CHARACTERS = ["'", " ", *list("abcdefghijklmnopqrstuvwxyz")]
BLANK = "_"
VOCAB = [BLANK] + CHARACTERS
char2idx = {c: i for i, c in enumerate(VOCAB)}
idx2char = {i: c for c, i in char2idx.items()}
vocab_list = VOCAB.copy()
vocab_list[0] = ""
beam_decoder = build_ctcdecoder(vocab_list)

device = torch.device("cpu")
total_infer_time = 0.0
num_infer_samples = 0

# === Audio Transforms ===

mel_transform = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=1024, hop_length=256)
db_transform = AmplitudeToDB()


# === Text Enrichment ===

def enrich_text(text):
    text = text.lower()
    music_keywords = ["stop", "pause", "resume", "continue", "play"]
    if any(kw in text for kw in music_keywords):
        text += " music"
    return text


# === Dataset Class ===

class SpeechDataset(Dataset):
    def __init__(self, data_dir, manifest_csv, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        with open(manifest_csv, newline='', encoding='utf-8') as f:
            for row in csv.reader(f):
                fname = os.path.basename(row[0])
                transcript = row[1].lower().strip()
                self.samples.append((fname, transcript))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, transcript = self.samples[idx]
        path = os.path.join(self.data_dir, fname)
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        spec = self.transform(waveform)
        spec = db_transform(spec)
        m = spec.squeeze(0).transpose(0, 1)
        target = torch.LongTensor([char2idx[c] for c in transcript if c in char2idx])
        return m, target


# === Collate Function ===

def collate_fn(batch):
    specs, targets = zip(*batch)
    input_lengths = torch.LongTensor([s.size(0) for s in specs])
    max_len = input_lengths.max().item()
    padded = torch.zeros(len(specs), max_len, N_MELS)
    for i, s in enumerate(specs):
        padded[i, :s.size(0), :] = s
    padded = padded.transpose(0, 1)
    target_lengths = torch.LongTensor([t.size(0) for t in targets])
    concatenated = torch.cat(targets)
    return padded, concatenated, input_lengths, target_lengths


# === ASR Model ===

class ASRModel(torch.nn.Module):
    def __init__(self, n_mels, hidden_dim, vocab_size, cnn_channels=64):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, cnn_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            torch.nn.BatchNorm2d(cnn_channels), torch.nn.ReLU(),
            torch.nn.Conv2d(cnn_channels, cnn_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            torch.nn.BatchNorm2d(cnn_channels), torch.nn.ReLU()
        )
        lstm_input = cnn_channels * (n_mels // 4)
        self.lstm = torch.nn.LSTM(lstm_input, hidden_dim, num_layers=4, bidirectional=True, dropout=0.3)
        self.layer_norm = torch.nn.LayerNorm(hidden_dim * 2)
        self.fc = torch.nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x = x.permute(1, 2, 0).unsqueeze(1)
        x = self.cnn(x)
        b, c, f, t = x.size()
        x = x.permute(3, 0, 1, 2).contiguous().view(t, b, c * f)
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        x = self.fc(x)
        return x.log_softmax(dim=-1)


# === Load Pretrained Classifiers ===

action_classifier = joblib.load("action_classifier.pkl")
object_classifier = joblib.load("object_classifier.pkl")

# ============================

train_df = pd.read_csv("train_labels.csv", header=None, names=["text", "action", "object"])
train_texts = train_df["text"].tolist()
train_actions = train_df["action"].map({0: "off", 1: "on"}).tolist()
train_objects = train_df["object"].tolist()

train_texts_enriched = [enrich_text(t) for t in train_texts]

# Train simple text-based classifiers
action_classifier = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", MultinomialNB())
])
action_classifier.fit(train_texts, train_actions)

object_classifier = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", MultinomialNB())
])
object_classifier.fit(train_texts_enriched, train_objects)


# === Audio Augmentation for Testing ===

def augment_audio(waveform, sample_rate):
    # Add light noise
    noise = torch.randn_like(waveform) * 0.002
    waveform += noise

    # Slight speed variation
    speed = 1.0
    if random.random() < 0.5:
        speed = random.choice([0.95, 1.05])
        new_sr = int(sample_rate * speed)
        waveform = torchaudio.functional.resample(waveform, sample_rate, new_sr)
        waveform = torchaudio.functional.resample(waveform, new_sr, sample_rate)

    return waveform



model = ASRModel(n_mels=N_MELS, hidden_dim=512, vocab_size=len(VOCAB)).to(device)
model.load_state_dict(torch.load("best_asr_model_last_best.pth", map_location=device))
model.eval()

test_ds = SpeechDataset("test_dataset", "test.csv", mel_transform)


# Apply augmentation on-the-fly via dataset wrapper

class AugmentedSpeechDataset(SpeechDataset):
    def __getitem__(self, idx):
        m, target = super().__getitem__(idx)
        path = os.path.join(self.data_dir, self.samples[idx][0])
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = augment_audio(waveform, SAMPLE_RATE)
        spec = self.transform(waveform)
        spec = db_transform(spec)
        m = spec.squeeze(0).transpose(0, 1)
        target = torch.LongTensor([char2idx[c] for c in self.samples[idx][1] if c in char2idx])
        return m, target


aug_test_ds = AugmentedSpeechDataset("test_dataset", "test.csv", mel_transform)
test_loader = DataLoader(aug_test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)


# === Beam Search Decoder ===

def beam_search_decode(logits, input_lens):
    logits = logits.cpu().permute(1, 0, 2)  # [Batch, Time, Vocab]
    results = []
    for b in range(logits.shape[0]):
        log_probs = logits[b, :input_lens[b]]
        decoded = beam_decoder.decode(log_probs.numpy())
        results.append(decoded.strip())
    return results


# === Inference Loop ===

test_refs, test_hyps = [], []
pred_actions, pred_objects = [], []

with torch.no_grad():
    for specs, targets, inp_l, tgt_l in tqdm(test_loader, desc="Testing"):
        specs = specs.to(device)
        inp_l = inp_l.to(device)

        start_time = time.time()
        logits = model(specs)
        end_time = time.time()

        total_infer_time += (end_time - start_time)
        num_infer_samples += specs.size(1)

        input_lengths_cnn = torch.full_like(inp_l, logits.size(0))
        hyps = beam_search_decode(logits.cpu(), input_lengths_cnn)

        offset = 0
        for l in tgt_l:
            ref = "".join(idx2char[i] for i in targets[offset:offset + l].tolist())
            test_refs.append(ref)
            offset += l

        test_hyps.extend(hyps)
        pred_actions.extend([action_classifier.predict([t])[0] for t in hyps])
        pred_objects.extend([object_classifier.predict([enrich_text(t)])[0] for t in hyps])


# === Metric Computation ===

def wer(r, h): return edit_distance(r.split(), h.split()) / max(1, len(r.split()))


def cer(r, h): return edit_distance(list(r), list(h)) / max(1, len(r))


total_wer = sum(wer(r, h) for r, h in zip(test_refs, test_hyps)) / len(test_refs)
total_cer = sum(cer(r, h) for r, h in zip(test_refs, test_hyps)) / len(test_refs)

print(f"\nTest WER: {total_wer:.3f}")
print(f"Test CER: {total_cer:.3f}")
print(f"Average inference time per sample: {total_infer_time / num_infer_samples:.4f} sec")

# === Action/Object Classification Accuracy ===

test_df = pd.read_csv("test_labels.csv", header=None, names=["text", "action", "object"])
test_actions = test_df["action"].map({0: "off", 1: "on"}).tolist()
test_objects = test_df["object"].tolist()

action_acc = accuracy_score(test_actions[:len(pred_actions)], pred_actions)
object_acc = accuracy_score(test_objects[:len(pred_objects)], pred_objects)

print(f"\nAction classification accuracy: {action_acc:.3f}")
print(f"Object classification accuracy: {object_acc:.3f}")
