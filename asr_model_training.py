import os
import csv
import torch
import torchaudio
import pandas as pd
import random
import joblib
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, FrequencyMasking, TimeMasking
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm
from torchaudio.functional import edit_distance
from pyctcdecode import build_ctcdecoder

# Constants and directories
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
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


# === Data Preparation and Classifier Training ===

def enrich_text(text):
    # Appends 'music' to text if it includes a music-related verb.
    text = text.lower()
    music_keywords = ["stop", "pause", "resume", "continue", "play"]
    if any(kw in text for kw in music_keywords):
        text += " music"
    return text


# Load and preprocess label data
train_df = pd.read_csv("train_labels.csv", header=None, names=["text", "action", "object"])
val_df = pd.read_csv("valid_labels.csv", header=None, names=["text", "action", "object"])

train_df = train_df.sample(frac=1).reset_index(drop=True)
val_df = val_df.sample(frac=1).reset_index(drop=True)

train_texts = train_df["text"].tolist()
train_actions = ["off" if a == 0 else "on" for a in train_df["action"]]
train_objects = train_df["object"].tolist()

val_texts = val_df["text"].tolist()
val_actions = ["off" if a == 0 else "on" for a in val_df["action"]]
val_objects = val_df["object"].tolist()

train_texts_enriched = [enrich_text(t) for t in train_texts]
val_texts_enriched = [enrich_text(t) for t in val_texts]

# Train Naive Bayes classifiers for action and object detection
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

# Save classifiers to disk
joblib.dump(action_classifier, "action_classifier.pkl")
joblib.dump(object_classifier, "object_classifier.pkl")

# === Audio Dataset and Loader

mel_transform = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=1024, hop_length=256)
db_transform = AmplitudeToDB()
freq_mask = FrequencyMasking(freq_mask_param=30)
time_mask = TimeMasking(time_mask_param=80)


class SpeechDataset(Dataset):
    def __init__(self, data_dir, manifest_csv, transform, augment=True):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        self.samples = []
        with open(manifest_csv, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                self.samples.append((os.path.basename(row[0]), row[1].lower().strip()))

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

        if self.augment:
            waveform += torch.randn_like(waveform) * 0.005
            speed_factor = random.choice([0.9, 1.1])
            new_sr = int(SAMPLE_RATE * speed_factor)
            waveform = torchaudio.functional.resample(waveform, SAMPLE_RATE, new_sr)
            waveform = torchaudio.functional.resample(waveform, new_sr, SAMPLE_RATE)

        spec = db_transform(self.transform(waveform))
        if self.augment:
            spec = freq_mask(spec)
            spec = time_mask(spec)

        m = spec.squeeze(0).transpose(0, 1)
        target = torch.LongTensor([char2idx[c] for c in transcript if c in char2idx])
        return m, target


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


# === ASR Model Definition ===

class ASRModel(nn.Module):
    def __init__(self, n_mels, hidden_dim, vocab_size, cnn_channels=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cnn_channels), nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cnn_channels), nn.ReLU()
        )
        self.lstm = nn.LSTM(cnn_channels * (n_mels // 4), hidden_dim, num_layers=4, bidirectional=True, dropout=0.3)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x = x.permute(1, 2, 0).unsqueeze(1)  # [B, 1, N_MELS, Time]
        x = self.cnn(x)
        b, c, f, t = x.size()
        x = x.permute(3, 0, 1, 2).contiguous().view(t, b, c * f)
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        return self.fc(x).log_softmax(dim=-1)


# Decode using beam search

def beam_search_decode(logits, input_lens):
    logits = logits.cpu().permute(1, 0, 2)
    results = []
    for b in range(logits.shape[0]):
        log_probs = logits[b, :input_lens[b]]
        decoded = beam_decoder.decode(log_probs.numpy())
        results.append(decoded.strip())
    return results


# === Prepare Datasets and DataLoaders ===

train_ds = SpeechDataset("train_dataset", "train.csv", mel_transform, augment=True)
val_ds = SpeechDataset("valid_dataset", "valid.csv", mel_transform, augment=False)
test_ds = SpeechDataset("test_dataset", "test.csv", mel_transform, augment=False)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)


# === Model Training Loop ===

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASRModel(N_MELS, hidden_dim=512, vocab_size=len(VOCAB)).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
ctc_loss = nn.CTCLoss(blank=char2idx[BLANK], zero_infinity=True)

NUM_EPOCHS = 100
patience = 7
best_val_loss = float('inf')
epochs_no_improve = 0

train_losses, val_losses, val_wers, val_cers = [], [], [], []

# Start training
for ep in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0
    for specs, targets, inp_l, tgt_l in tqdm(train_loader, desc=f"Train {ep}"):
        specs, targets = specs.to(device), targets.to(device)
        inp_l, tgt_l = inp_l.to(device), tgt_l.to(device)
        optimizer.zero_grad()
        logits = model(specs)
        input_lengths_cnn = torch.full_like(inp_l, logits.size(0))
        loss = ctc_loss(logits, targets, input_lengths_cnn, tgt_l)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss / len(train_loader))

    # Save validation metrics and decoded predictions
    model.eval()
    val_loss, val_wer, val_cer = 0, 0, 0
    predicted_actions_all, predicted_objects_all = [], []
    total_samples = 0
    refs = []

    with torch.no_grad():
        for specs, targets, inp_l, tgt_l in tqdm(val_loader, desc=f"Val {ep}"):
            specs, targets = specs.to(device), targets.to(device)
            inp_l, tgt_l = inp_l.to(device), tgt_l.to(device)
            logits = model(specs)
            input_lengths_cnn = torch.full_like(inp_l, logits.size(0))
            val_loss += ctc_loss(logits, targets, input_lengths_cnn, tgt_l).item()
            hyps = beam_search_decode(logits.cpu(), input_lengths_cnn)

            offset = 0
            for l in tgt_l:
                ref = "".join(idx2char[i] for i in targets[offset:offset + l].tolist())
                refs.append(ref)
                offset += l

            for r, h in zip(refs[-len(hyps):], hyps):
                val_wer += edit_distance(r.split(), h.split()) / max(1, len(r.split()))
                val_cer += edit_distance(list(r), list(h)) / max(1, len(r))
                total_samples += 1

                act_pred = action_classifier.predict([h])[0]
                obj_pred = object_classifier.predict([enrich_text(h)])[0]
                predicted_actions_all.append(act_pred)
                predicted_objects_all.append(obj_pred)

    val_losses.append(val_loss / len(val_loader))
    val_wers.append(val_wer / total_samples)
    val_cers.append(val_cer / total_samples)
    scheduler.step(val_losses[-1])

    print(
        f"Epoch {ep}: TrainLoss={train_losses[-1]:.4f}, ValLoss={val_losses[-1]:.4f}, WER={val_wers[-1]:.3f}, CER={val_cers[-1]:.3f}")

    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_asr_model.pth")
        print("*** Best model saved ***")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break


# === Export for Evaluation ===

# Export evaluation predictions and true labels for external evaluation script
torch.save({
    "val_actions_true": val_actions[:len(predicted_actions_all)],
    "val_actions_pred": predicted_actions_all,
    "val_objects_true": val_objects[:len(predicted_objects_all)],
    "val_objects_pred": predicted_objects_all,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "val_wers": val_wers,
    "val_cers": val_cers,
    "model": model,
    "test_loader": test_loader,
    "device": device
}, "eval_inputs.pth")
