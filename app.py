from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

clApp = ClientApp()

@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        message = request.get_json(force=True)
        encoded = message['image']
        decodeImage(encoded, clApp.filename)
        
        prediction = clApp.classifier.predict()
        
        # This sends {"prediction": "Stone"} to the JS success function
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)