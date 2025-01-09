# my_app/hubert_ser.py
import torch
import torch.nn.functional as F
import torchaudio

from transformers import AutoConfig, Wav2Vec2FeatureExtractor

# HuBERT-SER projesindeki modelleri içeri aktarma
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "HuBERT-SER"))
)
from models import HubertForSpeechClassification  # Doğru içeri aktarma

# Modeli global seviyede (uygulama start alırken) yükleyerek
# performanstan kazanabilirsiniz.
# Modelin ismini sizinkine göböylre düzenleyin.
MODEL_NAME_OR_PATH = "SeaBenSea/hubert-large-turkish-speech-emotion-recognition"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained(MODEL_NAME_OR_PATH)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME_OR_PATH)
sampling_rate = feature_extractor.sampling_rate
model = HubertForSpeechClassification.from_pretrained(MODEL_NAME_OR_PATH).to(device)


def speech_file_to_array_fn(file_path, sampling_rate):
    speech_array, original_sr = torchaudio.load(file_path)
    resampler = torchaudio.transforms.Resample(original_sr, sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def predict_emotion(file_path):
    # Tahmin fonksiyonu
    speech = speech_file_to_array_fn(file_path, sampling_rate)

    inputs = feature_extractor(
        speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True
    )
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model(**inputs).logits
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

    # Sonuçları düzenleyelim
    outputs = [
        {"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"}
        for i, score in enumerate(scores)
    ]
    return outputs
