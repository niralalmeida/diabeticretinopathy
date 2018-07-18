"""
 GUI for using Naive Bayes Classifier for predictions

 Date: 21st April, 18
"""

import warnings
warnings.filterwarnings('ignore')

import tkinter
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from PIL import Image, ImageTk
from sklearn.externals import joblib
from skimage.io import imread
from skimage.feature import hog

CURRENT_FILE = None

nb_lda = joblib.load('nb_lda.pkl')
gnb = joblib.load('gnb_pred.pkl')

rf_lda = joblib.load('rf_lda.pkl')
rfc = joblib.load('rfc_pred.pkl')

svm_lda = joblib.load('svm_lda.pkl')
svc = joblib.load('svm_pred.pkl')


def load_image():
    filename = askopenfilename()

    global CURRENT_FILE
    CURRENT_FILE = filename

    im = Image.open(filename)
    photo = ImageTk.PhotoImage(im)

    cv = tkinter.Canvas(width=512, height=512)
    cv.place(x=50, y=20)
    cv.create_image(10, 10, image=photo, anchor='nw')
    cv.image = im
    cv.photo = photo


def load_features(lda):
    img = imread(CURRENT_FILE)
    fd = hog(img, orientations=8, pixels_per_cell=(
        16, 16), cells_per_block=(1, 1), visualise=False)
    fd.shape = (1, -1)
    fd = lda.transform(fd)

    return fd


def pred_nb():
    if CURRENT_FILE is None:
        return

    fd = load_features(nb_lda)

    prediction = gnb.predict(fd)

    messagebox.showinfo('Naive Bayes Prediction',
                        'Predicted: {}'.format(prediction))


def pred_rfc():
    if CURRENT_FILE is None:
        return

    fd = load_features(rf_lda)

    prediction = rfc.predict(fd)

    messagebox.showinfo('Random Forests Prediction',
                        'Predicted: {}'.format(prediction))


def pred_svm():
    if CURRENT_FILE is None:
        return

    fd = load_features(svm_lda)

    prediction = svc.predict(fd)

    messagebox.showinfo('SVM Prediction', 'Predicted: {}'.format(prediction))


gui = tkinter.Tk()
gui.title('Diabetic Retinopathy Detection')

gui.minsize(600, 600)
gui.geometry("620x600")

load_button = tkinter.Button(
    gui, text="Load Image", command=load_image, height=2, width=10)
load_button.place(y=550, x=50)

predict_nb = tkinter.Button(
    gui, text="Predict NB", command=pred_nb, height=2, width=10)
predict_nb.place(y=550, x=220)

predict_rfc = tkinter.Button(
    gui, text="Predict RFC", command=pred_rfc, height=2, width=10)
predict_rfc.place(y=550, x=320)

predict_svm = tkinter.Button(
    gui, text="Predict SVM", command=pred_svm, height=2, width=10)
predict_svm.place(y=550, x=420)

gui.mainloop()
