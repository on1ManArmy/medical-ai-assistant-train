import os
import sqlite3
import numpy as np
import joblib

from flask import *
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# ---------------- CONFIG ---------------- #

app = Flask(__name__)
app.secret_key = "deepcare_secret"

UPLOAD_FOLDER = "database/Uploaded"
DB_PATH = "database/DeepCareX.db"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------- LOAD MODELS ---------------- #

print("Loading models...")

ALZ_MODEL = load_model("models/Alzheimer/Alzheimer_CNN.h5", compile=False)
KIDNEY_MODEL = load_model("models/Kidney/kidney.h5", compile=False)

BC_MODEL = joblib.load("models/Breast_Cancer/breast_cancer.pkl")
DIA_MODEL = joblib.load("models/Diabetes/diab_xg1.pkl")
HEP_MODEL = joblib.load("models/Hepatitis/hep.pkl")

print("Models loaded successfully")


# ---------------- DATABASE ---------------- #

def insert_user(name,email,password):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO USER VALUES (?,?,?)",(name,email,password))
    conn.commit()
    conn.close()


def check_user(name,password):

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM USER WHERE NAME=? AND PASSWORD=?",
        (name,password)
    )

    user = cur.fetchone()

    conn.close()

    return user is not None


def insert_contact(name,email,contact,msg):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("INSERT INTO CONTACT VALUES (?,?,?,?)",(name,email,contact,msg))

    conn.commit()
    conn.close()


def insert_newsletter(email):

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("INSERT INTO NEWSLETTER VALUES (?)",(email,))

    conn.commit()
    conn.close()


# ---------------- IMAGE UTILS ---------------- #

def preprocess_image(path):

    img = load_img(path,target_size=(128,128))
    img = img_to_array(img)
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    return img


def save_image(img):

    filename = secure_filename(img.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"],filename)

    img.save(path)

    return path


# ---------------- HOME ---------------- #

@app.route("/")
def home():
    return render_template("home.html")


# ---------------- LOGIN ---------------- #

# ---------------- LOGIN ---------------- #

@app.route("/login", methods=["POST"])
def login():

    name = request.form.get("name")
    password = request.form.get("password")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM USER WHERE NAME=? AND PASSWORD=?",
        (name, password)
    )

    user = cur.fetchone()
    conn.close()

    if user:
        session["user"] = name
        flash("Login successful", "success")
        return redirect(url_for("Alzheimer"))

    flash("Invalid Credential, Please try again", "danger")
    return redirect(url_for("home"))


# ---------------- REGISTER ---------------- #

@app.route("/register", methods=["POST"])
def register():

    username = request.form.get("username")
    email = request.form.get("useremail")
    password = request.form.get("userpassword")
    confirm = request.form.get("confirm_userpassword")

    if password != confirm:
        flash("Passwords do not match", "danger")
        return redirect(url_for("home"))

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO USER (NAME, EMAIL, PASSWORD) VALUES (?, ?, ?)",
        (username, email, password)
    )

    conn.commit()
    conn.close()

    flash("Registration successful", "success")

    return redirect(url_for("home"))
# ---------------- LOGOUT ---------------- #

@app.route("/logout")
def logout():

    session.pop("user",None)

    return redirect(url_for("home"))


# ---------------- PAGES ---------------- #

@app.route("/Alzheimer")
def Alzheimer():

    if "user" in session:
        return render_template("Alzheimer.html")

    return redirect(url_for("home"))


@app.route("/Kidney")
def Kidney():

    if "user" in session:
        return render_template("Kidney.html")

    return redirect(url_for("home"))


@app.route("/Breast_Cancer")
def Breast_Cancer():

    if "user" in session:
        return render_template("Breast_Cancer.html")

    return redirect(url_for("home"))


@app.route("/Diabetes")
def Diabetes():

    if "user" in session:
        return render_template("Diabetes.html")

    return redirect(url_for("home"))


@app.route("/Hepatitis")
def Hepatitis():

    if "user" in session:
        return render_template("Hepatitis.html")

    return redirect(url_for("home"))


@app.route("/About")
def About():
    return render_template("About.html")


@app.route("/Contact")
def Contact():
    return render_template("Contact.html")


# ---------------- CONTACT FORM ---------------- #

@app.route("/Reply",methods=["POST"])
def Reply():

    name = request.form["name"]
    email = request.form["email"]
    contact = request.form["contact"]
    msg = request.form["message"]

    insert_contact(name,email,contact,msg)

    flash("Message sent successfully")

    return redirect(url_for("Contact"))


# ---------------- NEWSLETTER ---------------- #

@app.route("/Newsletter",methods=["POST"])
def Newsletter():

    email = request.form["email"]

    insert_newsletter(email)

    flash("Subscribed successfully")

    return redirect(url_for("home"))


# ---------------- ALZHEIMER REPORT ---------------- #

@app.route("/Alzheimer_Report",methods=["POST"])
def Alzheimer_Report():

    name = request.form["name"]
    pid = request.form["id"]
    age = request.form["age"]
    gender = request.form["gender"]

    img = request.files["image"]

    path = save_image(img)

    img = preprocess_image(path)

    pred = ALZ_MODEL.predict(img)

    classes = [
        "Mild Demented",
        "Moderate Demented",
        "Non Demented",
        "Very Mild Demented"
    ]

    result = classes[np.argmax(pred)]

    return render_template(
        "output.html",
        disease="Alzheimer",
        result=result,
        name=name,
        pid=pid,
        age=age,
        gender=gender
    )
# ---------------- KIDNEY REPORT ---------------- #

@app.route("/Kidney_Report",methods=["POST"])
def Kidney_Report():

    name = request.form["name"]
    pid = request.form["id"]
    age = request.form["age"]
    gender = request.form["gender"]

    img = request.files["image"]

    path = save_image(img)

    img = preprocess_image(path)

    pred = KIDNEY_MODEL.predict(img)

    classes = [
        "Kidney Cyst",
        "Normal",
        "Kidney Stone",
        "Kidney Tumor"
    ]

    result = classes[np.argmax(pred)]

    return render_template(
        "output.html",
        disease="Kidney Disease",
        result=result,
        name=name,
        pid=pid,
        age=age,
        gender=gender
    )


# ---------------- BREAST CANCER ---------------- #

@app.route("/Breast_Cancer_Report",methods=["POST"])
def Breast_Cancer_Report():

    name = request.form["name"]
    pid = request.form["id"]
    age = request.form["age"]
    gender = request.form["gender"]

    radius = float(request.form["radius"])
    texture = float(request.form["texture"])
    perimeter = float(request.form["perimeter"])
    area = float(request.form["area"])
    smoothness = float(request.form["smoothness"])

    inp = np.array([[radius,texture,perimeter,area,smoothness]])

    pred = BC_MODEL.predict(inp)[0]

    result = "Malignant" if pred==1 else "Benign"

    return render_template(
        "output.html",
        disease="Breast Cancer",
        result=result,
        name=name,
        pid=pid,
        age=age,
        gender=gender
    )


# ---------------- DIABETES ---------------- #

@app.route("/Diabetes_Report",methods=["POST"])
def Diabetes_Report():

    name = request.form["name"]
    pid = request.form["id"]
    age = request.form["age"]
    gender = request.form["gender"]

    bmi = float(request.form["bmi"])
    glucose = float(request.form["glucose"])
    insulin = float(request.form["insulin"])

    inp = np.array([[age,bmi,glucose,insulin]])

    pred = DIA_MODEL.predict(inp)[0]

    result = "Diabetic" if pred==1 else "Non Diabetic"

    return render_template(
        "output.html",
        disease="Diabetes",
        result=result,
        name=name,
        pid=pid,
        age=age,
        gender=gender
    )


# ---------------- HEPATITIS ---------------- #

@app.route("/Hepatitis_Report",methods=["POST"])
def Hepatitis_Report():

    name = request.form["name"]
    pid = request.form["id"]
    age = request.form["age"]
    gender = request.form["gender"]

    alb = float(request.form["ALB"])
    alp = float(request.form["ALP"])
    alt = float(request.form["ALT"])
    ast = float(request.form["AST"])

    inp = np.array([[age,alb,alp,alt,ast]])

    pred = HEP_MODEL.predict(inp)[0]

    result = "Hepatitis Detected" if pred==1 else "No Hepatitis"

    return render_template(
        "output.html",
        disease="Hepatitis",
        result=result,
        name=name,
        pid=pid,
        age=age,
        gender=gender
    )


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True)