import whisper
import sounddevice as sd
import numpy as np
import threading
import queue
import time
from datetime import datetime

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

# ////////////////////////////////////////////////////
# Testing: Uncomment to test basic transcription, and comment out rest
# ////////////////////////////////////////////////////
# Start live transcription (set duration as needed)
# transcribe_live(duration=5)
# while True:
#     transcribe_live(duration=CHUNK_DURATION)





def format_timestamp(timestamp):
    """Converts a timestamp into a readable format (HH:MM:SS.sss)."""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%H:%M:%S.%f")[:-3]  # Drop microseconds to milliseconds

def record_audio():
    """Continuously records audio and pushes chunks into the queue."""
    def callback(indata, frames, stream_time, status):
        """Captures audio and timestamps it."""
        if status:
            print(f"Recording error: {status}")
        timestamp = time.time()  # Use time.time() to get the current timestamp
        audio_queue.put((timestamp, indata.copy()))  # Add timestamp with audio chunk

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,  # Mono audio
        callback=callback,
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
        # dtype='float32'
    ):
        print("Recording... Press Ctrl+C to stop.")
        while True:
            pass

def process_audio():
    """Processes audio chunks from the queue using Whisper."""
    while True:
        if not audio_queue.empty():
            timestamp, audio_chunk = audio_queue.get()
            formatted_time = format_timestamp(timestamp)
            print(f"Processing chunk recorded at: {formatted_time}")  # Debug information
            audio_queue.task_done()  # Mark as processed

            # Convert to Whisper-compatible format
            audio_data = np.squeeze(audio_chunk).astype(np.float32)
            audio_data = audio_data / np.abs(audio_data).max()  # Normalize to [-1.0, 1.0]

            # Transcribe using Whisper
            try:
                result = model.transcribe(audio_data, fp16=False)
                print(f"Transcription ({formatted_time}): {result['text']}")
            except Exception as e:
                print(f"Error during transcription: {e}")

# Create threads
recording_thread = threading.Thread(target=record_audio, daemon=True)
processing_thread = threading.Thread(target=process_audio, daemon=True)

# Start threads
recording_thread.start()
processing_thread.start()

# Keep main thread running
try:
    while True:
        pass
except KeyboardInterrupt:
    print("Stopping...")
