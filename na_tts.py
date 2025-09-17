import asyncio
import edge_tts
import tempfile
import os
import sys
import shutil
import subprocess

class TTTS:
    def __init__(self, voice="en-US-AriaNeural"):
        self.voice = voice

        # Make sure ffplay is on PATH
        if not shutil.which("ffplay"):
            raise RuntimeError("ffplay not found in PATH—please install ffmpeg.")

    async def speak(self, text: str, pitch: str = "+20Hz"):
        """
        pitch: use Hz (e.g. '+20Hz', '-10Hz') or semitones ('+2st', '-3st').
        """
        # Directly tell edge-tts to shift pitch
        communicate = edge_tts.Communicate(
            text,
            self.voice,
            pitch=pitch  # ← here!
        )

        # write to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            tmp_mp3 = tf.name

        await communicate.save(tmp_mp3)

        try:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_mp3],
                check=True
            )
        finally:
            os.remove(tmp_mp3)

async def main():
    tts = TTTS("en-US-AriaNeural")
    await tts.speak("Now Edge TTS works without realtime streaming!", pitch="+30Hz")

if __name__ == "__main__":
    asyncio.run(main())
