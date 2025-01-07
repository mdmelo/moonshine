import sys
import time
from enum import Enum

from sounddevice import query_devices
import speech_recognition as sr
import numpy as np

from gtts import gTTS
from pydub import AudioSegment
from pygame import mixer
from io import BytesIO
from queue import Queue
from moonif import moonshine_main_live_audio

import pygame._sdl2 as sdl2


class TaskOp(Enum):
    none = 0
    add = 1
    delete = 2
    show = 3


tasks = []
taskstate = TaskOp.none

_debug = True


def get_play_devices():
    mixer.init()
    devices = sdl2.audio.get_audio_device_names(False)
    return devices


def speak(text, lang='en', play=True, delay=2):
    fp = BytesIO()
    tts = gTTS(text, lang=lang)
    tts.write_to_fp(fp)

    if play:
        # mixer.init()
        fp.seek(0)
        mixer.music.load(fp, "mp3")

        # see https://www.pygame.org/docs/ref/music.html#pygame.mixer.music.set_volume
        # The volume argument is a float between 0.0 and 1.0 that sets the volume level.
        # When new music is loaded the volume is reset to full volume. If volume is a negative
        # value it will be ignored and the volume will remain set at the current level. If the
        # volume argument is greater than 1.0, the volume will be set to 1.0.
        mixer.music.set_volume(1.0)
        mixer.music.play()

    if delay > 0:
        time.sleep(delay)

    return fp


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


def respond(response_text, delay=2):
    if _debug:
        print("TTS>>> ", response_text)
    speak(response_text, lang='en', play=True, delay=delay)


def main():
    global tasks
    global taskstate

    triggerKeyword = "wahoo"
    command = ""

    while command != triggerKeyword:
        command = listen_for_command()

    while True:
        command = listen_for_command()

        if command:
            if taskstate == TaskOp.add:
                tasks.append(command)
                taskstate = TaskOp.none
                respond("Adding " + command + " to your task list. You have " + str(len(tasks)) + " currently in your list.")

            elif "add a task" in command:
                taskstate = TaskOp.add
                respond("What is the task?")

            elif "list tasks" in command:
                respond("Your tasks are:")
                for task in tasks:
                    respond(task)

            elif "exit" in command:
                respond("Goodbye!")
                break

            else:
                respond("Sorry, I'm not sure how to handle that command.")


def process_command(text):
    global tasks
    global taskstate

    if _debug:
        print("\nMoonshine VR supplies '{}'".format(text))

    # consider using distance calc for word matching
    # see /media/mike/MMBKUPDRV/yottaa/Books/MachineLearning/Voice/voice-control/spoken-command-processor/processor/utils.py

    if text:
        command = text.casefold()

        if taskstate == TaskOp.add:
            tasks.append(command)
            respond("Sure, adding " + command + " to your task list", 3)
            respond("You have " + str(len(tasks)) + " tasks now.", 3)
            taskstate = TaskOp.none

        elif taskstate == TaskOp.delete:
            tasks.remove(command)
            respond("Sure, removing " + command + " from your task list", 3)
            respond("You have " + str(len(tasks)) + " tasks now.", 3)
            taskstate = TaskOp.none

        elif "add task" in command:
            taskstate = TaskOp.add
            respond("Sure, what is the task?")

        elif "list task" in command:
            respond("Sure, your tasks are:")
            for task in tasks:
                respond(task)

        elif "delete task" in command:
            taskstate = TaskOp.delete
            respond("Sure, what is the task?")

        elif "exit" in command:
            respond("Goodbye!")
            sys.exit(0)

        else:
            respond("Sorry, I'm not sure how to handle that command.", 2)


if __name__ == "__main__":
    # ['Built-in Audio Analog Stereo', 'Plantronics C720-M Analog Stereo']
    # devices = get_play_devices()
    playdevice = "Built-in Audio Analog Stereo"
    mixer.pre_init(devicename=playdevice)
    mixer.init()

    recdevice = "Plantronics C720-M"
    devices = query_devices()
    for ndx, d in enumerate(devices):
        if d["name"].startswith(recdevice):
            break

    print("play device {}, record device {} [{}]".format(playdevice, ndx, d["name"]))

    # test text-to-speech
    sound = speak('hello, Welcome to Voice Assistant!')
    time.sleep(5)

    respond("Greetings, Wahoo!")

    moonshine_main_live_audio(cbfunc=process_command, model_name="moonshine/base", device=ndx)
    respond("Goodbye, Wahoo!")

    mixer.quit()
