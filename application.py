import os
import warnings

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from Algorithm import Algorithm
from werkzeug.utils import secure_filename

# Suppress the specific FutureWarning from the transformers library
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with your actual secret key

# Initialize the Algorithm class
algorithm = Algorithm()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file uploaded", 400

        text = file.read().decode("utf-8")
        fig = algorithm.plot_emotion_graph(text, progress_callback_chunk=None, progress_callback_overall=None)

        # Ensure the static directory exists
        if not os.path.exists("static"):
            os.makedirs("static")

        fig_path = os.path.join("static", "graph.png")
        fig.savefig(fig_path)
        return render_template("result.html", graph_image="static/graph.png")

    return render_template("index.html")

@app.route("/download-graph")
def download_graph():
    return send_file("static/graph.png", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Use this for local testing

    # Uncomment the following line for production deployment using waitress
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=8080)
