import asyncio
import sounddevice as sd
import numpy as np
import time
from faster_whisper import WhisperModel
import os
import threading
import keyboard  # pip install keyboard

class FastWhisperRecognizer:
    def __init__(self, model_size="small.en", device=None, sample_rate=16000):
        import torch
        if device is None:
            device = "cpu"
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        self.sample_rate = sample_rate
        self.recording = False
        self.audio = []

    def _callback(self, indata, frames, time, status):
        if self.recording:
            self.audio.append(indata.copy())




    def _listen_blocking(self, ptt_key="*"):
        print(f"Push and hold '{ptt_key}' to talk...")
        while True:
            keyboard.wait(ptt_key)
            print("🎙️ Recording...")
            self.audio = []
            self.recording = True

            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                callback=self._callback
            ):
                while keyboard.is_pressed(ptt_key):
                    time.sleep(0.05)

            self.recording = False
            print("🛑 Stopped recording. Transcribing...")

            audio_data = np.concatenate(self.audio).flatten()
            if len(audio_data) == 0:
                print("No audio recorded, try again.")
                continue

            segments, _ = self.model.transcribe(audio_data, language="en", beam_size=5)
            text = " ".join([seg.text for seg in segments]).strip()
            print("Recognized:", text)
            if text:
                return text

    async def listen_push_to_talk(self, ptt_key="*"):
        # Run the blocking method in a background thread safely
        return await asyncio.to_thread(self._listen_blocking, ptt_key)

    def transcribe_file(self, file_path):
        print(f"Transcribing from file: {file_path}")
        try:
            segments, _ = self.model.transcribe(file_path, beam_size=5)
            text = " ".join([seg.text for seg in segments]).strip()
            print("Recognized:", text)
        finally:
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"⚠️ Failed to delete file {file_path}: {e}")
        return text