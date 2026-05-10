import whisper
import os

class LyricsEngine:
    def __init__(self, model_size="base"):
        print(f"Loading Whisper model ({model_size})...")
        # Note: This requires ffmpeg to be installed on the system
        self.model = whisper.load_model(model_size)

    def extract_lyrics(self, audio_path):
        print(f"Transcribing audio: {audio_path}")
        # fp16=False ensures it runs on CPU if no GPU is available
        result = self.model.transcribe(audio_path, verbose=False, fp16=False)
        
        segments = []
        for segment in result['segments']:
            segments.append({
                "start": segment['start'],
                "end": segment['end'],
                "text": segment['text'].strip()
            })
        
        print(f"Detected {len(segments)} lyric segments.")
        return segments

    @staticmethod
    def get_current_lyric(segments, t):
        for segment in segments:
            if segment['start'] <= t <= segment['end']:
                return segment['text']
        return ""
