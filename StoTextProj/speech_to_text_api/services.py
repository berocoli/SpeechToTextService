import os
from django.conf import settings
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
from pydub.silence import split_on_silence
from textblob import TextBlob
from googletrans import Translator
import azure.cognitiveservices.speech as speechsdk
import logging

logger = logging.getLogger(__name__)


def cut_whitenoise(
    audio_path, output_path, silence_thresh=-40, min_silence_len=500, keep_silence=200
):
    """
    Ses dosyasındaki sessiz alanları kaldırır ve yeni bir dosya oluşturur.
    """
    try:
        audio = AudioSegment.from_file(audio_path, format="wav")
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence,
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


def save_audio_segments(audio_path, segments, output_dir):
    """
    Konuşmacı segmentlerini ayrı ses dosyalarına kaydeder.
    """
    try:
        audio = AudioSegment.from_file(audio_path, format="wav")
        segment_paths = []

        for idx, (start, end, speaker) in enumerate(segments, start=1):
            segment_audio = audio[start * 1000 : end * 1000]  # Milisaniyeye çevir
            segment_path = os.path.join(output_dir, f"{speaker}_{idx}.wav")
            segment_audio.export(segment_path, format="wav")
            segment_paths.append(segment_path)

        return segment_paths
    except Exception as e:
        logger.error(f"Segment kaydedilirken hata oluştu: {e}")
        return []


def diarize_audio(audio_path):
    """
    Pyannote.audio ile konuşmacı ayrımı yapar ve tutarlı konuşmacı etiketleri oluşturur.
    """
    hf_token = settings.HUGGINGFACE_TOKEN
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    diarization_result = pipeline(audio_path)

    # Create a mapping of original speaker labels to normalized ones
    speaker_mapping = {}
    speaker_count = 0
    segments = []

    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        if speaker not in speaker_mapping:
            speaker_mapping[speaker] = f"SPEAKER_{speaker_count:02d}"
            speaker_count += 1

        normalized_speaker = speaker_mapping[speaker]
        segments.append((turn.start, turn.end, normalized_speaker))

    # Sort segments by start time
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
            "subjectivity": blob.sentiment.subjectivity,
        }
    except Exception as e:
        return {"error": str(e)}


def transcribe_with_azure(segment_wav_path):
    """
    Belirtilen .wav dosyasını Azure Speech ile metne dönüştürür.
    """
    speech_config = speechsdk.SpeechConfig(
        subscription=settings.AZURE_API_KEY, region=settings.AZURE_REGION
    )
    speech_config.speech_recognition_language = "tr-TR"

    audio_input = speechsdk.AudioConfig(filename=segment_wav_path)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_input
    )

    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return "[NoMatch: Ses anlaşılamadı]"
    elif result.reason == speechsdk.ResultReason.Canceled:
        return f"[Canceled: {result.cancellation_details.reason}]"
