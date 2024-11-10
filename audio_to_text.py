import speech_recognition as sr
from pyAudioAnalysis import audioSegmentation as aS

def perform_diarization(audio_file):
    # Perform speaker diarization (2 speakers)
    [segments, classes] = aS.speaker_diarization(audio_file, 2, plot_res=True)
    return segments, classes

def transcribe_segment(segment, audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source, offset=segment[0], duration=segment[1] - segment[0])
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return "Error with the recognition service"

def diarize_and_transcribe(audio_file):
    segments, classes = perform_diarization(audio_file)
    
    # Iterate through the diarization segments
    for i, segment in enumerate(segments):
        speaker = classes[i]
        text = transcribe_segment(segment, audio_file)
        print(f"Speaker {speaker} said: {text}")

# Example usage
audio_file_path = "recording.aac"  # Replace with your audio file path
diarize_and_transcribe(audio_file_path)
