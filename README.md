<h1>Gym Back Machine Detector</h1>

![Alt text](./media/ezgif-5-81ce1448c1.gif)

A gym back machine detector using datasets of images from Google images, Youtube videos, TikTok, Bilibili, Roboflow and Kaggle datasets.

<h3>Built With</h3>

+ [Python](https://www.python.org/downloads/)
+ [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
+ [CometML](https://github.com/comet-ml)
+ [Flask](https://github.com/pallets/flask)
+ [Containerization with Docker](https://docs.docker.com/get-started/overview/)
<!-- + [Amazon AWS deployment](https://aws.amazon.com/)
+ [CI/CD with Github Actions](https://github.com/features/actions) -->
+ [AnyLabeling](https://anylabeling.nrl.ai/)
  
 <h3>About the Dataset</h3>

The dataset consists of 7 back gym Machines. Although there was some off-the-shelf dataset usage, most of the images were manually collected throughout the internet due to the unavailability of a decent sample size for most of the machines. However, whenever a dataset that already includes one of the classes is found, the class is extracted from the dataset and added to the training of the model.

|         | Excercises | Machine seated mid row <sup>{0}<sup> | Seated cable row <sup>{1}<sup> | Cable lat pulldown <sup>{2}<sup> | Machine lat pulldown <sup>{3}<sup> | Chest supported T-bar row <sup>{4}<sup> | Landmine T-bar row <sup>{5}<sup> | Machine reverse fly <sup>{6}<sup> |
|---------|------------|-------------------------------|------------------|-------------------|----------------------|--------------------------|--------------------|----------------------|
| Dataset | Train      | 511                           | 712              | 989               | 376                  | 352                      | 331                | 425                  |
|         | Val        | 49                            | 69               | 83                | 30                   | 30                       | 32                 | 41                   |
|         | Test       | 50                            | 147              | 57                | 44                   | 61                       | 87                 | 55                   |
| Total   |            | 610                           | 928              | 1129              | 450                  | 443                      | 450                | 521                  |

## How to run

+ STEP 01- Clone the repository

```bash
git clone https://github.com/dahshury/Back-Machine-Detector
```

+ STEP 02- Create a conda environment after opening the repository (optional)

```bash
conda create -n gymback python=3.9 -y
```

```bash
conda activate gymback
```

+ STEP 03- install the requirements

```bash
pip install -r requirements.txt
```

+ STEP 04- training requirements using GPU (optional)

```bash
pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


### To Run prediction using the pretrained model

run the following command in the terminal

```bash
python app.py
```

Now, open the following link in the browser

```bash
 0.0.0.0:8080
```

You can now upload an image of a desired category, and click predict for the result.

### Results

| Model                                                                                 | mAP<sup>val<br>50 | mAP<sup>val<br>50-95 | Test Accuracy |
| ------------------------------------------------------------------------------------- | ----------------- | -------------------- | ------------- |
| YOLOv8 nano (from scratch)                                                          | 0.95399           | 0.74986              | ?         |
| [YOLOv8 medium](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 0.96725           | 0.80451              | 0.84         |

<h5>Pretrained (medium) model results on video:</h5>

https://github.com/Masterx3/Back-Machine-Detector/assets/83991104/efc1ff09-4964-42e5-bb87-f7638a85f25a

+ For nano model validation metrics, check [[1]](./runs/detect/train2/).

+ For medium pre-trained model validation metrics, check [[2]](./runs/detect/train/).

### Contact Me

[Linkedin](https://www.linkedin.com/in/dahshory/)

### Acknowledgements

Thanks to [Samir Gouda](github.com/SamirGouda) & [Omar Eldahshoury](github.com/omareldahshoury) for thier support.
