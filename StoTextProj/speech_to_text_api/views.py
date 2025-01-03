from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
import os
from django.conf import settings
from .services import cut_whitenoise, diarize_audio, transcribe_with_azure, analyze_sentiment_turkish
import uuid

def home(request):
    return render(request, "home.html")

def upload_audio(request):
    """
    Kullanıcıdan ses dosyasını alır ve seçilen işleme türüne göre işler.
    """
    if request.method == "POST":
        # Kullanıcıdan dosya ve işlem türünü al
        audio_file = request.FILES.get("audio_file")
        process_type = request.POST.get("process_type")  # "clean" veya "raw"

        if not audio_file:
            return JsonResponse({"error": "Dosya yüklenmedi."}, status=400)

        # Dosyayı kaydetmek için benzersiz bir ad oluştur
        audio_id = uuid.uuid4().hex
        original_audio_path = os.path.join(settings.MEDIA_ROOT, f"{audio_id}_original.wav")
        cleaned_audio_path = os.path.join(settings.MEDIA_ROOT, f"{audio_id}_cleaned.wav")

        # Dosyayı diske kaydet
        with open(original_audio_path, "wb+") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        # İşleme türüne göre dosyayı temizle veya olduğu gibi kullan
        if process_type == "clean":
            processed_audio_path = cut_whitenoise(original_audio_path, cleaned_audio_path)
            if not processed_audio_path:
                return JsonResponse({"error": "Dosya temizleme işlemi başarısız."}, status=500)
        else:
            processed_audio_path = original_audio_path

        # Konuşmacı ayrımı
        hf_token = "YOUR_HF_TOKEN"  # Hugging Face Token
        diarized_segments = diarize_audio(processed_audio_path, hf_token)

        # Speech-to-Text ve Duygu Analizi
        speech_config = "YOUR_AZURE_API_CONFIG"
        results = []
        for start, end, speaker in diarized_segments:
            segment_audio_path = f"/tmp/{audio_id}_{speaker}.wav"

            # Azure Speech-to-Text
            transcript = transcribe_with_azure(segment_audio_path, speech_config)
            sentiment = analyze_sentiment_turkish(transcript)

            # Sonuçları listeye ekle
            results.append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "transcript": transcript,
                "sentiment": sentiment
            })

        return JsonResponse({"results": results}, status=200)

    return render(request, "upload_audio.html")
