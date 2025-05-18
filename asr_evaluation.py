# === Imports ===
import os
import csv
import time
import random
import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch import nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score
)

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

# === Audio Preprocessing Transforms ===

SAMPLE_RATE = 16000
N_MELS = 128
mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=1024, hop_length=256)
db_transform = torchaudio.transforms.AmplitudeToDB()
freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=30)
time_mask = torchaudio.transforms.TimeMasking(time_mask_param=80)

# === Dataset Definition for ASR ===

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
        return m, torch.tensor([])  # target is unused during evaluation

# === Collate Function for Test DataLoader ===

def collate_fn(batch):
    specs, _ = zip(*batch)
    input_lengths = torch.LongTensor([s.size(0) for s in specs])
    max_len = input_lengths.max().item()
    padded = torch.zeros(len(specs), max_len, N_MELS)
    for i, s in enumerate(specs):
        padded[i, :s.size(0), :] = s
    padded = padded.transpose(0, 1)
    return padded, None

# === Create Directory for Evaluation Results ===

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Load Saved Evaluation Inputs ===

data = torch.load("eval_inputs.pth", weights_only=False)

val_actions_true = data["val_actions_true"]
val_actions_pred = data["val_actions_pred"]
val_objects_true = data["val_objects_true"]
val_objects_pred = data["val_objects_pred"]
train_losses = data["train_losses"]
val_losses = data["val_losses"]
val_wers = data["val_wers"]
val_cers = data["val_cers"]
model = data["model"]
test_loader = data["test_loader"]
device = data["device"]

# === Evaluation: Classification Metrics ===

def evaluate_classification(true_labels, predicted_labels, label_type="Action", labels_order=None):
    acc = accuracy_score(true_labels, predicted_labels)
    prec = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    rec = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

    print(f"\n {label_type.upper()} Classification:")
    print(f"   Accuracy:  {acc:.3f}")
    print(f"   Precision: {prec:.3f}")
    print(f"   Recall:    {rec:.3f}")
    print(f"   F1-score:  {f1:.3f}")

# === Plot Training Loss and WER/CER Curves ===

def plot_training_curves(train_losses, val_losses, val_wers, val_cers):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Validation Loss")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/loss_curve.png")
    plt.close()

    plt.plot(val_wers, label='WER')
    plt.plot(val_cers, label='CER')
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate")
    plt.title("WER / CER per Epoch")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/wer_cer_curve.png")
    plt.close()

# === Measure Average Inference Latency ===

def measure_average_latency(model, sample_batch, device):
    latencies = []
    model.eval()
    with torch.no_grad():
        for i in range(sample_batch.size(1)):
            input_tensor = sample_batch[:, i:i+1, :].to(device)
            start = time.time()
            _ = model(input_tensor)
            end = time.time()
            latencies.append(end - start)
    avg_latency = sum(latencies) / len(latencies)
    print(f"\n Average inference latency: {avg_latency:.3f} seconds")
    return avg_latency

# === Run Evaluation Procedures ===

evaluate_classification(val_actions_true, val_actions_pred, label_type="Action", labels_order=["off", "on"])
evaluate_classification(val_objects_true, val_objects_pred, label_type="Object", labels_order=sorted(set(val_objects_true)))

plot_training_curves(train_losses, val_losses, val_wers, val_cers)
sample_batch = next(iter(test_loader))[0]
measure_average_latency(model, sample_batch=sample_batch, device=device)
