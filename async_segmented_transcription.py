import os
import sys
import sounddevice as sd
import numpy as np
import wave
import whisper
import signal
import argparse

# Ours
import TextCleaner

# Constants
AUDIO_DIR = "audio_segments"
TEXT_DIR = "text_transcriptions"
SAMPLE_RATE = 16000
CHANNELS = 1  # Mono audio

# Editable configurations
CLIP_DURATION = 10  # in seconds

# Signal handler for clean exit during recording
recording = True


def signal_handler(sig, frame):
    global recording
    recording = False
    print("\nRecording stopped.")


signal.signal(signal.SIGINT, signal_handler)


def get_device_index(device_name):
    """
    Get the index of the audio input device by its name.
    """
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if device_name.lower() in device['name'].lower():
            return index
    return None


def record_audio(use_blackhole=False):
    """
    Record audio in 1-minute clips and save to WAV files.
    """
    device_name = "blackhole" if use_blackhole else "built-in"
    device_index = get_device_index(device_name)
    if device_index is None:
        print(f"Error: Audio input device '{device_name}' not found.")
        sys.exit(1)

    os.makedirs(AUDIO_DIR, exist_ok=True)
    clip_index = len(os.listdir(AUDIO_DIR))  # Start naming files from the existing count

    print(f"Recording {CLIP_DURATION} seconds audio clips using {device_name} microphone. Press Ctrl+C to stop.")
    while recording:
        audio_data = sd.rec(
            int(CLIP_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16',
            device=device_index
        )
        sd.wait()  # Wait until recording is finished

        # Save audio to a file
        filename = os.path.join(AUDIO_DIR, f"clip_{clip_index:04d}.wav")
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 2 bytes per sample
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

        print(f"Saved: {filename}")
        clip_index += 1

cleaner = TextCleaner.TextCleaner()
def process_audio():
    """
    Process WAV files into text transcriptions using Whisper.
    """
    os.makedirs(TEXT_DIR, exist_ok=True)
    model = whisper.load_model("base.en", device="cpu")  # Use Whisper's base model

    audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')])
    if not audio_files:
        print(f"No audio files found in '{AUDIO_DIR}'. Record audio first.")
        return

    print(f"Processing audio clips to generate text transcriptions...")
    for audio_file in audio_files:
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        print(f"Transcribing: {audio_path}")
        result = model.transcribe(audio_path)

        # Clean Text with text Cleaner
        text = cleaner.clean_text(result['text'])

        # Save the transcription to a text file
        text_filename = os.path.join(TEXT_DIR, f"{os.path.splitext(audio_file)[0]}.txt")
        with open(text_filename, 'w') as text_file:
            text_file.write(text)
        
        print(f"Saved transcription: {text_filename}")


def main():
    parser = argparse.ArgumentParser(description="Segmented audio transcription script.")
    parser.add_argument("mode", choices=["record", "process"], help="Mode: 'record' to record audio, 'process' to transcribe audio")
    parser.add_argument("device", nargs="?", default="default", choices=["default", "virtual"],
                        help="Device: 'default' for built-in microphone, 'virtual' for BlackHole microphone (default: 'default')")

    args = parser.parse_args()

    if args.mode == "record":
        use_blackhole = args.device == "virtual"
        record_audio(use_blackhole=use_blackhole)
    elif args.mode == "process":
        process_audio()


if __name__ == "__main__":
    main()
