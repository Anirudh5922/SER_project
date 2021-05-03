import librosa
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#ALL emotions in RAVDESS dataset

def convert(audio_path):
    """
    This function will convert any .wav files into .wav files compatible
    with the model. -ac audio channels 1 (monochannel), -ar audio frequency 44100hz

    audio_path (str): the path associated with the .wav file that will be converted
    """
    if not os.path.exists(audio_path):
	       return "File Doesn't Exist"
    file_split_list = audio_path.split("/")
    filename = file_split_list[-1].split(".")[0]
    new_filename = f"{filename}_converted.wav"
    file_split_list[-1] = new_filename
    seperator = "/"
    target_path = seperator.join(file_split_list)
    if not audio_path.endswith(".wav"):
        return "Invalid File: Must be in .wav format"
    else:
        try:
            os.system(f"ffmpeg -i {audio_path} -ac 1 -ar 44100 {target_path}")
        except Exception as e:
            print(e)
            return
        return target_path
