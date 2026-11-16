
from flask import Flask, render_template, request

from src.predict import predict_disease, encoder
from src.disease_information import get_description, get_precautions

app = Flask(__name__)

# Get all available symptoms from the trained symptom encoder
AVAILABLE_SYMPTOMS = sorted(encoder.classes_)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    prediction = None
    description = None
    precautions = []
    selected_symptoms = []
    error = None

    if request.method == "POST":

        selected_symptoms = request.form.getlist("symptoms")

        if not selected_symptoms:
            error = "Please select at least one symptom."

        else:
            try:
                # Call the existing trained ML model
                prediction = predict_disease(selected_symptoms)

                # Retrieve disease information
                description = get_description(prediction)
                precautions = get_precautions(prediction)

            except Exception as exception:
                error = f"Prediction could not be completed: {exception}"

    return render_template(
        "predictor.html",
        symptoms=AVAILABLE_SYMPTOMS,
        selected_symptoms=selected_symptoms,
        prediction=prediction,
        description=description,
        precautions=precautions,
        error=error
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
