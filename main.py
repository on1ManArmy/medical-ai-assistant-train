import os
import sqlite3
import numpy as np

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ---------------- APP CONFIG ---------------- #

app = Flask(__name__)
app.secret_key = "deepcarex_secret"

UPLOAD_FOLDER = "database/Uploaded"
DB_PATH = "database/DeepCareX.db"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOAD MODELS ---------------- #

print("Loading AI Models...")

MODELS = {
    "alz": load_model("Models/Alzheimer/Alzheimer_CNN.hdf5", compile=False),
    "brain": load_model("Models/Brain_Tumor/Brain_Tumor_VGG19.hdf5", compile=False),
    "covid": load_model("Models/COVID/Covid.hdf5", compile=False),
    "pneumonia": load_model("Models/Pneumonia/Pneumonia_DenseNet201.hdf5", compile=False),
    "kidney": load_model("Models/Kidney/Kidney.hdf5", compile=False)
}

print("All models loaded successfully!")

# ---------------- DATABASE ---------------- #

def execute(query, params=()):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(query, params)
    conn.commit()
    conn.close()

def fetch(query, params=()):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(query, params)
    data = cur.fetchall()
    conn.close()
    return data

# ---------------- IMAGE HELPERS ---------------- #

def save_image(img):
    filename = secure_filename(img.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    img.save(path)
    return path

def preprocess_image(path, size=(128,128)):
    img = load_img(path, target_size=size)
    img = img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(model, path):

    img = preprocess_image(path)
    pred = model.predict(img)

    return np.argmax(pred), float(np.max(pred))

# ---------------- AUTH ---------------- #

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login",methods=["POST"])
def login():

    name = request.form["name"]
    password = request.form["password"]

    user = fetch(
        "SELECT * FROM USER WHERE NAME=? AND PASSWORD=?",
        (name,password)
    )

    if user:
        session["user"] = name
        return render_template("Alzheimer.html",login_user=name)

    flash("Invalid credentials")
    return redirect(url_for("home"))

@app.route("/signup",methods=["POST"])
def signup():

    name = request.form["username"]
    email = request.form["email"]
    password = request.form["password"]

    execute(
        "INSERT INTO USER VALUES(?,?,?)",
        (name,email,password)
    )

    session["user"]=name
    return render_template("Alzheimer.html",login_user=name)

@app.route("/logout")
def logout():
    session.pop("user",None)
    return redirect(url_for("home"))

# ---------------- PATIENT DATA ---------------- #

def get_patient():

    return {
        "name": request.form.get("name"),
        "id": request.form.get("id"),
        "age": request.form.get("age"),
        "gender": request.form.get("gender"),
    }

# ---------------- ALZHEIMER ---------------- #

@app.route("/Alzheimer_Report",methods=["POST"])
def Alzheimer_Report():

    data = get_patient()

    img = request.files["image"]
    img_path = save_image(img)

    pred,conf = predict_image(MODELS["alz"],img_path)

    classes = [
        "MildDemented",
        "ModerateDemented",
        "NonDemented",
        "VeryMildDemented"
    ]

    result = f"{classes[pred]} ({conf*100:.2f}%)"

    return render_template(
        "output.html",
        name=data["name"],
        id_=data["id"],
        age=data["age"],
        gender=data["gender"],
        disease="Alzheimer",
        result=result
    )

# ---------------- BRAIN TUMOR ---------------- #

@app.route("/Brain_Tumor_Report",methods=["POST"])
def Brain_Tumor_Report():

    data = get_patient()

    img = request.files["image"]
    img_path = save_image(img)

    pred,conf = predict_image(MODELS["brain"],img_path)

    classes = [
        "Glioma",
        "Meningioma",
        "No Tumor",
        "Pituitary"
    ]

    result = f"{classes[pred]} ({conf*100:.2f}%)"

    return render_template(
        "output.html",
        name=data["name"],
        id_=data["id"],
        age=data["age"],
        gender=data["gender"],
        disease="Brain Tumor",
        result=result
    )

# ---------------- PNEUMONIA ---------------- #

@app.route("/Pneumonia_Report",methods=["POST"])
def Pneumonia_Report():

    data = get_patient()

    img = request.files["image"]
    img_path = save_image(img)

    pred,conf = predict_image(MODELS["pneumonia"],img_path)

    classes=["Normal","Pneumonia"]

    result = f"{classes[pred]} ({conf*100:.2f}%)"

    return render_template(
        "output.html",
        name=data["name"],
        id_=data["id"],
        age=data["age"],
        gender=data["gender"],
        disease="Pneumonia",
        result=result
    )

# ---------------- COVID ---------------- #

@app.route("/Covid_Report",methods=["POST"])
def Covid_Report():

    data = get_patient()

    img = request.files["image"]
    img_path = save_image(img)

    pred,conf = predict_image(MODELS["covid"],img_path)

    classes=["COVID Positive","COVID Negative"]

    result = f"{classes[pred]} ({conf*100:.2f}%)"

    return render_template(
        "output.html",
        name=data["name"],
        id_=data["id"],
        age=data["age"],
        gender=data["gender"],
        disease="COVID",
        result=result
    )

# ---------------- KIDNEY ---------------- #

@app.route("/Kidney_Report",methods=["POST"])
def Kidney_Report():

    data = get_patient()

    img = request.files["image"]
    img_path = save_image(img)

    pred,conf = predict_image(MODELS["kidney"],img_path)

    classes=[
        "Kidney Cyst",
        "Normal",
        "Kidney Stone",
        "Kidney Tumor"
    ]

    result = f"{classes[pred]} ({conf*100:.2f}%)"

    return render_template(
        "output.html",
        name=data["name"],
        id_=data["id"],
        age=data["age"],
        gender=data["gender"],
        disease="Kidney Disease",
        result=result
    )

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True,port=5001)