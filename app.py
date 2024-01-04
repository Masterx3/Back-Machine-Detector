import os
from ultralytics import YOLO
from GymDetector.pipeline.training_pipeline import TrainingPipeline
from GymDetector.utils.main_utils import decode_img, encode_img
from flask import Flask, request, jsonify, render_template, Response
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
        # get image
        image = request.json['image']
        
        # decode base64 image to image and save it in ./data as clApp.filename
        decode_img(image, clApp.filename)
        
        # instantiate the model
        model = YOLO('runs/detect/train/weights/best.pt')
        
        # inference
        results = model.predict(source='data/inputImg.jpg', save=True)
        
        # phone app results
        phone_result ={}
        
        for i, result in enumerate(results):
            
            phone_result[f'class_of_instance_{i}'] = int(result.boxes.cls[0])  # Extract and convert class to integer
            phone_result[f'conf_of_instance_{i}'] = float(result.boxes.conf[0])  # Extract and convert confidence to float

            # Extract xyxy coordinates as a list of floats
            xyxy_list = [float(coord) for coord in result.boxes.xyxy[0].tolist()]
            phone_result[f'xyxy_of_instance_{i}'] = xyxy_list
            
        # web app results
        opencodedbase64 = encode_img("runs/detect/predict/inputImg.jpg")
        web_result = {"image": opencodedbase64.decode('utf-8')}
        
        os.system('rm -rf runs/detect/predict')
        
    except ValueError as val:
        print(val)
        return Response('Value not found inside JSON data')
    
    except KeyError:
        return Response('Key value error incorrect key passed')
    
    except Exception as e:
        print(e)
        result = f'invalid_input, {e}'
        
    client_type = request.args.get('client_type', default='web')
    if client_type == 'phone':
        return jsonify(phone_result)
    else:
        return jsonify(web_result)

if __name__ == '__main__':
    clApp = ClientApp()
    # app.run(host=APP_HOST, port=APP_PORT)
    app.run(host='localhost', port=80) # for Azure