from flask import Flask, request, render_template
from Algorithm import Algorithm  # Import your Algorithm class
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)
algorithm = Algorithm()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    fig = algorithm.plot_emotion_graph(text)

    # Save the plot to a bytes buffer and convert to a base64 string
    buf = io.BytesIO()
    FigureCanvas(fig).print_png(buf)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('ascii')

    return render_template('result.html', image_base64=image_base64)


if __name__ == "__main__":
    app.run(debug=True)
