import threading
from flask import Flask, render_template, request, redirect, url_for, flash
from Algorithm import Algorithm  # Import your Algorithm class

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with your secret key

algorithm = Algorithm()

@app.route('/')
def home():
    return render_template('index.html')

def analyze_text(text):
    # Perform the emotion analysis
    fig = algorithm.plot_emotion_graph(text)
    fig.savefig('static/graph.png')  # Save the graph to a file
    # Notify the user that the analysis is complete (you can use flash or another method)

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            text = file.read().decode('utf-8')
            # Start the analysis in a new thread
            analysis_thread = threading.Thread(target=analyze_text, args=(text,))
            analysis_thread.start()
            flash("File uploaded successfully. The analysis is in progress.")
            return redirect(url_for('loading_screen'))

@app.route('/loading')
def loading_screen():
    return render_template('loading.html')

@app.route('/result')
def result_screen():
    # Render the result page where the user can see the graph
    return render_template('result.html', graph='static/graph.png')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
