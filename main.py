from Bard import Chatbot
from playsound import playsound
import speech_recognition as sr
from os import system
import whisper
import warnings
import sys

token = 'PASTE YOUR TOKEN HERE'
chatbot = Chatbot(token)
r = sr.Recognizer()

tiny_model = whisper.load_model('tiny')
base_model = whisper.load_model('base')
warnings.filterwarnings("ignore", message="FP16 is not support on CPU, using FP32 instead.")

if sys.platform != 'darwin':
    import pyttsx3
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate-50)

def prompt_bard(prompt):
    response = chatbot.ask(prompt)
    return response['content']

def speak(text):
    if sys.platform == 'darwin':
        ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-_$: ")
        cleaned_text = ''.join(c for c in text if c in ALLOWED_CHARS)
        system(f'say "{cleaned_text}"')
    else:
        engine.say(text)
        engine.runAndWait()

def main():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        while True:
            while True:
                try:
                    print('\nSay "google" to wake me up. \n')
                    audio = r.listen(source)
                    with open("wake_detect.wav", "wb") as f:
                        f.write(audio.get_wav_data())
                    result = tiny_model.transcribe("wake_detect.wav")
                    text_input = result['text']
                    if 'google' in text_input.lower().strip():
                        break
                    else:
                        print('Sorry, no wake word found. Try again.')
                except Exception as e:
                    print("Error transcribing audio: ", e)
                    continue