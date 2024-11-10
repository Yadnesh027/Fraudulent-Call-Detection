import sounddevice as sd
import whisper
import numpy as np
import tempfile
import queue
import torch
from pyannote.audio import Pipeline
import wave

# Initialize Whisper model (small model for better speed)
model = whisper.load_model("small")

# Load pyannote speaker diarization model
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")

# Queue for storing audio chunks    
audio_queue = queue.Queue()

# Callback function to capture audio in real time
def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

# Real-time audio recording parameters
sample_rate = 16000
channels = 1

# Function to save queue audio to a temporary WAV file for processing
def save_audio_to_file(queue, filename):
    frames = []
    while not queue.empty():
        frames.append(queue.get())
    audio_data = np.concatenate(frames, axis=0)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

# Function to perform diarization and transcription
def transcribe_and_diarize():
    while True:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_wav:
            # Save audio from queue to a file for processing
            save_audio_to_file(audio_queue, tmp_wav.name)

            # Perform diarization
            diarization_result = pipeline({"audio": tmp_wav.name})
            
            # Transcribe each segment
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                with tempfile.NamedTemporaryFile(suffix=".wav") as segment_wav:
                    save_audio_to_file(audio_queue, segment_wav.name)
                    result = model.transcribe(segment_wav.name)
                    print(f"Speaker {speaker}: {result['text']}")

# Start recording audio in the background
with sd.InputStream(callback=audio_callback, channels=channels, samplerate=sample_rate):
    print("Recording... Press Ctrl+C to stop.")
    try:
        transcribe_and_diarize()
    except KeyboardInterrupt:
        print("\nStopped recording.")
