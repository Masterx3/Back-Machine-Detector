import os
import sys
from GymDetector.pipeline.training_pipeline import TrainingPipeline
from GymDetector.utils.main_utils import decode_img, encode_img
from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS, cross_origin
from GymDetector.constant.application import APP_HOST, APP_PORT

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = 'inputImg.jpg'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train')
def train_route():
    obj = TrainingPipeline()
    obj.run_pipeline()
    return "training successful."

@app.route('/predict', methods=['POST', 'GET'])
@cross_origin()
def predict_route():
    try:
        image = request.json['image']
        decode_img(image, clApp.filename)
        
        os.system("yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.25 source=data/inputImg.png save=true")
        opencodedbase64 = encode_img("runs/detect/predict/inputImg.png")
        result = {"image": opencodedbase64.decode('utf-8')}
        os.system('rm -rf runs/detect/predict')
        
    except ValueError as val:
        print(val)
        return Response('Value not found inside JSON data')
    
    except KeyError:
        return Response('Key value error incorrect key passed')
    
    except Exception as e:
        print(e)
        result = 'invalid_input'
        
    return jsonify(result)

if __name__ == '__main__':
    clApp = ClientApp()
    app.run(host=APP_HOST, port=APP_PORT)
    app.run(host='0.0.0.0', port=80) # for Azure