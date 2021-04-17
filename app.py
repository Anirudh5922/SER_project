from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from utils import extract_feature,convert
import librosa
import pickle
import os
import keras
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
app = Flask(__name__)

## Configure upload location for audio
app.config['UPLOAD_FOLDER'] = "./audio"

## Route for home page
@app.route('/')
def home():
    return render_template('index.html',value="")


## Route for results
@app.route('/results', methods = ['GET', 'POST'])
def results():
    """
    This route is used to save the file, convert the audio to 16000hz monochannel,
    and predict the emotion using the saved binary model
    """
    if not os.path.isdir("./audio"):
        os.mkdir("audio")
    if request.method == 'POST':
        try:
          f = request.files['file']
          filename = secure_filename(f.filename)
          f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
          print("File Saved "+filename)
        except:
          return render_template('index.html', value="")

    wav_file_pre  = os.listdir("./audio")[0]
    wav_file_pre = f"{os.getcwd()}\\audio\\{wav_file_pre}"
    wav_file = convert(wav_file_pre)
    check,sr = librosa.load(wav_file_pre ,res_type='kaiser_fast')
    print(wav_file_pre)
    os.remove(wav_file_pre)
    mfcc_exmp = np.mean(librosa.feature.mfcc(y=check, sr=sr, n_mfcc=40).T,axis=0)
    model=keras.models.load_model('hinds2_model.h5')
    observed_emotions = ['neutral','angry','happy','fearful','sad']
    mfcc_exmp = np.expand_dims(mfcc_exmp, axis=0)
    mfcc_exmp = np.expand_dims(mfcc_exmp, axis=2)
    pred = model.predict_classes(mfcc_exmp)
    print(pred)
    #model = pickle.load(open(f"{os.getcwd()}/model.model", "rb"))
#    x_test =extract_feature(wav_file)
#    y_pred=model.predict(np.array([x_test]))
    #os.remove(wav_file_pre)
    return render_template('index.html', value=observed_emotions[pred[0]])#y_pred[0])
    print(y_pred)