import whisper
import sounddevice as sd
import numpy as np

import os
import ssl
import urllib.request

# For now doing this to bypass ssl issues TODO
ssl._create_default_https_context = ssl._create_unverified_context

# Load the Whisper model
model = whisper.load_model("base.en", device="cpu")  # Change to "tiny" if needed
# model = whisper.load_model("tiny.en", device="cpu")

# Audio settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # seconds
# TODO: Implment a queue and 2 threads so one continuously produces/records audio, and other process it
QUEUE_MAX_SIZE = 10 

# Shared queue
audio_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)

# Function to capture audio and transcribe it
def transcribe_live(duration=5, sample_rate=SAMPLE_RATE):
    print("Recording... Speak now!")

    # Record audio for the specified duration
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is complete

    # Convert to 16-bit PCM format as required by Whisper
    # audio = (audio * 32767).astype(np.int16)

    # Transcribe audio using Whisper
    print("Transcribing...")
    result = model.transcribe(audio=audio.flatten(), language="en")
    print("Transcription:", result['text'])

# Start live transcription (set duration as needed)
# transcribe_live(duration=5)
while True:
    transcribe_live(duration=CHUNK_DURATION)