import speech_recognition as sr
import threading

recognizer = sr.Recognizer()

# Function to process recognized audio
def process_audio(audio):
    try:
        # Recognize the speech using Google Web API
        text = recognizer.recognize_google(audio)
        # Print the recognized text immediately (without blocking the listening thread)
        print(f"Recognized Text: {text}")
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError:
        print("Network error occurred")

# Continuous listening function
def listen_continuously():
    with sr.Microphone() as source:
        print("Listening... (Press Ctrl+C to stop)")
        recognizer.adjust_for_ambient_noise(source)
        
        while True:
            try:
                # Listen for a phrase (with phrase_time_limit set)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)  # Adjust timeout and phrase_time_limit as needed
                print("Processing...")

                # Start a new thread to process the audio while listening for the next phrase
                threading.Thread(target=process_audio, args=(audio,)).start()

            except sr.WaitTimeoutError:
                print("No speech detected, waiting for next input...")
            except KeyboardInterrupt:
                print("\nStopped listening.")
                break

# Start the continuous listening
listen_continuously()
