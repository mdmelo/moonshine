import speech_recognition as sr
import numpy as np

from gtts import gTTS
from pydub import AudioSegment
from pygame import mixer
from io import BytesIO
from queue import Queue
from moonif import moonshine_main_live_audio

import pyautogui
import time



_debug = True

def speak(text, lang='en', play=True, delay=0):
    fp = BytesIO()
    tts = gTTS(text, lang=lang)
    tts.write_to_fp(fp)

    if play:
        mixer.init()
        fp.seek(0)
        mixer.music.load(fp, "mp3")
        mixer.music.play()

    if delay > 0:
        time.sleep(delay)

    return fp


# test text-to-speech
sound = speak('hello, Welcome to Voice Assistant!')
time.sleep(5)


def listen_for_command():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening for commands...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print("You said:", command)
        return command.lower()
    except sr.UnknownValueError:
        print("Could not understand audio. Please try again.")
        return None
    except sr.RequestError:
        print("Unable to access the Google Speech Recognition API.")
        return None

def respond(response_text, delay=0):
    if _debug: print("TTS>>> ", response_text)
    speak(response_text, lang='en', play=True, delay=delay)

tasks = []
listeningToTask = False




def main():
    global tasks
    global listeningToTask

    while True:
        command = listen_for_command()

        # XXX FIX ME
        triggerKeyword = "wahoo"

        if command and triggerKeyword in command:
            if listeningToTask:
                tasks.append(command)
                listeningToTask = False
                respond("Adding " + command + " to your task list. You have " + str(len(tasks)) + " currently in your list.")
            elif "add a task" in command:
                listeningToTask = True
                respond("Sure, what is the task?")
            elif "list tasks" in command:
                respond("Sure. Your tasks are:")
                for task in tasks:
                    respond(task)
            elif "take a screenshot" in command:
                pyautogui.screenshot("screenshot.png")
                respond("I took a screenshot for you.")
            elif "open chrome" in command:
                respond("Opening Chrome.")
                webbrowser.open("http://www.youtube.com/@JakeEh")
            elif "exit" in command:
                respond("Goodbye!")
                break
            else:
                respond("Sorry, I'm not sure how to handle that command.")


def process_command(text):
    print("\nMoonshine VR supplies '{}'".format(text))

    # consider using distance calc for word matching
    # see /media/mike/MMBKUPDRV/yottaa/Books/MachineLearning/Voice/voice-control/spoken-command-processor/processor/utils.py


if __name__ == "__main__":
    respond("Greetings, Wahoo!", 2)

    moonshine_main_live_audio(cbfunc=process_command, model_name="moonshine/base")
    respond("Goodbye, Wahoo!", 2)


