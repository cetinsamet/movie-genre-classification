import sys
import numpy as np

from Preprocessor import Preprocessor
from OneHotEncoder import OneHotEncoder
from Vectorizer import Vectorizer
from Model import Model

from keras.models import model_from_json

def load_keras_model(model_path='../model/'):
    with open(model_path +"model.json", 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path+"model.h5")
    return loaded_model

def main(argv):

    if len(argv) != 1:
        print("Usage: python2 predict_genre.py input-subtitle-path")
        exit()

    p = Preprocessor()      # Preprocessor is initialized
    v = Vectorizer()        # Vectorizer is initialized
    v.load()                # Vectorizer is loaded
    enc = OneHotEncoder()   # Encoder is initialized
    enc.load()              # Encoder is loaded

    # READ INPUT SUBTITLE AND PREPROCESS
    subtitle_path           = argv[0]
    subtitle_text           = ' '.join([line for line in open(subtitle_path, 'r')]).replace('\n', ' ')
    subtitle_text_processed = p.preprocess_document(subtitle_text)

    # VECTORIZE PREPROCESSED SUBTITLE
    X   = v._transform(subtitle_text_processed, method='count')
    X   = np.reshape(X, (1, -1))

    model       = load_keras_model()        # Load model
    pred_onehot = model.predict(X)          # Make prediction
    genre       = enc.decode(pred_onehot)   # Encode one-hot label
    print("Predicted Genre: %s" % genre)

    return

if __name__ == '__main__':
    main(sys.argv[1:])