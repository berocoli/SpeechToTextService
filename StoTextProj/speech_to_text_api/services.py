import os
from django.conf import settings
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
from pydub.silence import split_on_silence
from textblob import TextBlob
from googletrans import Translator
import azure.cognitiveservices.speech as speechsdk


def cut_whitenoise(audio_path, output_path, silence_thresh=-40, min_silence_len=500, keep_silence=200):
    """
    Ses dosyasındaki sessiz alanları kaldırır ve yeni bir dosya oluşturur.
    """
    try:
        audio = AudioSegment.from_file(audio_path, format="wav")
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )

        if not chunks:
            return None

        cleaned_audio = AudioSegment.silent(duration=0)
        for chunk in chunks:
            cleaned_audio += chunk

        cleaned_audio.export(output_path, format="wav")
        return output_path

    except Exception as e:
        return None
    
def diarize_audio(audio_path, hf_token):
    """
    Pyannote.audio ile konuşmacı ayrımı yapar.
    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    diarization_result = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))

    segments.sort(key=lambda x: x[0])
    return segments

def analyze_sentiment_turkish(text):
    """
    Metni İngilizce'ye çevirip duygu analizi yapar.
    """
    translator = Translator()
    try:
        translated_text = translator.translate(text, src="tr", dest="en").text
        blob = TextBlob(translated_text)
        return {
            "original_text": text,
            "translated_text": translated_text,
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
    except Exception as e:
        return {"error": str(e)}

def transcribe_with_azure(segment_wav_path, speech_config):
    """
    Belirtilen .wav dosyasını Azure Speech ile metne dönüştürür.
    """
    audio_input = speechsdk.AudioConfig(filename=segment_wav_path)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_input
    )

    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return "[NoMatch: Ses anlaşılamadı]"
    elif result.reason == speechsdk.ResultReason.Canceled:
        return f"[Canceled: {result.cancellation_details.reason}]"
