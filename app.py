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
app.config['UPLOAD_FOLDER'] = "./static/audio"

## Route for home page
@app.route('/')
def home():
    return render_template('index0.html')


## Route for results
@app.route('/results/<lang>', methods = ['GET', 'POST'])
def results(lang):
    """
    This route is used to save the file, convert the audio to 16000hz monochannel,
    and predict the emotion using the saved binary model
    """
    if not os.path.isdir("./static/audio"):
        os.mkdir("./static/audio")
    if request.method == 'GET':
        return render_template('index.html', value="No Audio",f="",l=lang)
    if request.method == 'POST':
        try:
          if len(os.listdir("./static/audio"))!=0:
            wav_file_pre  = os.listdir("./static/audio")[0]
            wav_file_pre = f"{os.getcwd()}\\static\\audio\\{wav_file_pre}"
            os.remove(wav_file_pre)

          f = request.files['file']
          filename = secure_filename(f.filename)
          f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
          print("File Saved "+filename)
        except:
          return render_template('index.html', value="No Audio",f="",l=lang)

    wav_file_name  = os.listdir("./static/audio")[0]
    wav_file_pre = f"{os.getcwd()}\\static\\audio\\{wav_file_name}"

    wav_file = convert(wav_file_pre)
    check,sr = librosa.load(wav_file_pre ,res_type='kaiser_fast')
    print(wav_file_pre)
    mfcc_exmp = np.mean(librosa.feature.mfcc(y=check, sr=sr, n_mfcc=40).T,axis=0)
    if lang=="English":
       model=keras.models.load_model('ravdess_tess_savee_speech.h5')
       observed_emotions = ['Neutral','Calm','Happy','Sad','Angry','Fearful','Disgust','Surprised']
    else:
       model=keras.models.load_model('hinds2_model.h5')
       observed_emotions = ['Neutral','Angry','Happy','Fearful','Sad']
    mfcc_exmp = np.expand_dims(mfcc_exmp, axis=0)
    mfcc_exmp = np.expand_dims(mfcc_exmp, axis=2)
    pred = model.predict_classes(mfcc_exmp)
    print(pred)
    #model = pickle.load(open(f"{os.getcwd()}/model.model", "rb"))
#    x_test =extract_feature(wav_file)
#    y_pred=model.predict(np.array([x_test]))
    #os.remove(wav_file_pre)
    return render_template('index.html', value=observed_emotions[pred[0]],f=filename,l=lang)#y_pred[0])
    print(y_pred)
