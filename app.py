from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load metrics
with open("metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    cluster = None
    if request.method == "POST":
        income = float(request.form["income"])
        score = float(request.form["score"])
        cluster = model.predict([[income, score]])[0]

    return render_template(
        "index.html",
        cluster=cluster,
        silhouette=round(metrics["silhouette"], 3)
    )

if __name__ == "__main__":
    app.run(debug=True)
