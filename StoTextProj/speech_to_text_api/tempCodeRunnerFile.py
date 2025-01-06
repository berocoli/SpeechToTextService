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
