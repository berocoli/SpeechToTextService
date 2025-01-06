import logging
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.conf import settings
import os
import uuid
from .services import (
    cut_whitenoise,
    diarize_audio,
    transcribe_with_azure,
    analyze_sentiment_turkish,
    save_audio_segments,
)

from .speechEmotion import predict_emotion

# Logger'ı oluştur
logger = logging.getLogger(__name__)

# Proje kök dizinini manuel olarak tanımlayın
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def home(request):
    """
    Ana sayfa
    """
    return render(request, "home.html")


def emotion(request):
    """
    Duygu analizi sayfası
    """
    return render(request, "emotion.html")


def about(request):
    """
    Hakkında sayfası
    """
    return render(request, "about.html")


def contact(request):
    """
    İletişim sayfası
    """
    return render(request, "contact.html")


def upload_audio(request):
    """
    Kullanıcıdan ses dosyasını alır, işler ve sonuçları bir .txt dosyası olarak döndürür.
    """
    if request.method == "POST":
        logger.info("POST isteği alındı. Ses dosyası işleniyor...")

        # Kullanıcıdan dosya ve işlem türünü al
        audio_file = request.FILES.get("audio_file")
        process_type = request.POST.get("process_type")  # "clean" veya "raw"
        logger.info(f"İşleme türü: {process_type}")

        if not audio_file:
            logger.error("Dosya yüklenmedi.")
            return JsonResponse({"error": "Dosya yüklenmedi."}, status=400)

        # dosya adları
        audio_id = uuid.uuid4().hex
        original_audio_path = os.path.join(BASE_DIR, f"{audio_id}_original.wav")
        cleaned_audio_path = os.path.join(BASE_DIR, f"{audio_id}_cleaned.wav")

        # diske kaydet
        try:
            with open(original_audio_path, "wb+") as f:
                for chunk in audio_file.chunks():
                    f.write(chunk)
            logger.info(f"Ses dosyası kaydedildi: {original_audio_path}")
        except Exception as e:
            logger.error(f"Dosya kaydedilirken bir hata oluştu: {e}")
            return JsonResponse({"error": "Dosya kaydedilemedi."}, status=500)

        # İşleme türüne göre dosyayı temizle veya olduğu gibi kullan
        if process_type == "clean":
            processed_audio_path = cut_whitenoise(
                original_audio_path, cleaned_audio_path
            )
            if not processed_audio_path:
                logger.error(f"Dosya temizleme işlemi başarısız.")
                return JsonResponse(
                    {"error": "Dosya temizleme işlemi başarısız."}, status=500
                )
            logger.info(f"Temizlenmiş dosya: {processed_audio_path}")
        else:
            processed_audio_path = original_audio_path

        # Konuşmacı ayrımı ve segmentlere ayırma
        try:
            diarized_segments = diarize_audio(processed_audio_path)
            segment_paths = save_audio_segments(
                processed_audio_path, diarized_segments, BASE_DIR
            )
            logger.info(f"{len(segment_paths)} segment başarıyla kaydedildi.")
        except Exception as e:
            logger.error(f"Konuşmacı ayrımı sırasında hata oluştu: {e}")
            return JsonResponse({"error": "Konuşmacı ayrımı başarısız."}, status=500)

        # Sonuçları işleyip .txt dosyasına yazma
        dialogue = []
        current_speaker = None
        for idx, segment_path in enumerate(segment_paths, start=1):
            try:
                transcript = transcribe_with_azure(segment_path)
                sentiment = analyze_sentiment_turkish(transcript)
                sentiment_polarity = (
                    "Olumlu"
                    if sentiment["polarity"] > 0
                    else "Olumsuz" if sentiment["polarity"] < 0 else "Nötr"
                )

                # Konuşmacı etiketi ve metni oluştur
                speaker_label = os.path.basename(segment_path).split("_")[
                    0:2
                ]  # Konuşmacı etiketlerini birleştirme
                speaker_label = " ".join(speaker_label)

                # Konuşmacı değiştiyse yeni bir satır oluştur
                if speaker_label != current_speaker:
                    dialogue.append(f"\n{speaker_label}:\n")
                    current_speaker = speaker_label

                dialogue.append(f"  {transcript} " f"(Duygu: {sentiment_polarity})\n")
            except Exception as e:
                logger.error(f"Segment işleme hatası: {e}")
                dialogue.append(
                    f"{os.path.basename(segment_path)}: Hata: İşleme hatası.\n"
                )

        # .txt dosyası oluştur
        txt_file_path = os.path.join(BASE_DIR, f"{audio_id}_dialogue.txt")
        try:
            with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                txt_file.writelines(dialogue)
            logger.info(f".txt dosyası oluşturuldu: {txt_file_path}")
        except Exception as e:
            logger.error(f".txt dosyası oluşturulurken hata oluştu: {e}")
            return JsonResponse({"error": "Sonuç dosyası oluşturulamadı."}, status=500)
        finally:
            # Geçici dosyaları temizle
            for path in [original_audio_path, cleaned_audio_path, *segment_paths]:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Geçici dosya silindi: {path}")

        # .txt dosyasını kullanıcıya indirt
        with open(txt_file_path, "r", encoding="utf-8") as txt_file:
            response = HttpResponse(txt_file.read(), content_type="text/plain")
            response["Content-Disposition"] = (
                f'attachment; filename="{audio_id}_dialogue.txt"'
            )
            return response

    return render(request, "upload_audio.html")


def emotion_analysis(request):
    if request.method == "POST":
        audio_file = request.FILES.get("audio")
        if not audio_file:
            return JsonResponse({"error": "No audio file provided"}, status=400)

        # Geçici bir path’e kaydetme
        temp_path = os.path.join(settings.MEDIA_ROOT, audio_file.name)
        with open(temp_path, "wb+") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        # Fonksiyonu çağırıyoruz
        results = predict_emotion(temp_path)

        return JsonResponse({"results": results}, status=200)
    else:
        return JsonResponse({"error": "Only POST requests allowed"}, status=405)


def test_endpoint(request):
    data = {"message": "URL Çalışıyor!"}
    return JsonResponse(data)
