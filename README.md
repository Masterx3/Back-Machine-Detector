<h1>Gym Back Machine Detector</h1>

![Alt text](./media/ezgif-5-81ce1448c1.gif)

A gym back machine detector using datasets of images from Google images, Youtube videos, TikTok, Bilibili, Roboflow and Kaggle datasets. The following models have been trained:

+ [YOLOv8-nano](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt)
+ [YOLOv8-medium](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt)
+ [RT-DETRv2-X](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth)

<h3>Built With</h3>

+ [Python](https://www.python.org/downloads/)
+ [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
+ [ONNX](https://onnx.ai/)
+ [CometML](https://github.com/comet-ml)
+ [Flask](https://github.com/pallets/flask)
+ [Streamlit](https://streamlit.io/)
+ [Containerization with Docker](https://docs.docker.com/get-started/overview/)
+ [Microsoft Azure deployment](https://azure.microsoft.com/)
+ [CI/CD with Github Actions](https://github.com/features/actions)
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

+ STEP 04- install training requirements using GPU (optional)

```bash
pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### To Run prediction using the pretrained model

run the following command in the terminal

```bash
python app2.py
```

Alternatively, you can launch the static HTML webpage

```bash
python app.py
```

Now, open the following link in the browser

```bash
localhost:8080
```

You can now upload an image of a desired category, and click predict for the result.

### Results

| Model                                                                                 | mAP<sup>val<br>50 | mAP<sup>val<br>50-95 | mAP<sup>test<br>50 | mAP<sup>test<br>50-95 |
| ------------------------------------------------------------------------------------- | ----------------- | -------------------- | ------------------ | --------------------- |
| [YOLOv8-nano (from-scratch)](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt)                                                          | 0.95399           | 0.74986              | 0.848              | 0.604                   |
| [YOLOv8-medium (finetuned)](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt) | 0.96725           | 0.80451              | 0.874              | 0.683                 |
| [RT-DETRv2-X (finetuned)](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth)                                                         | 0.972             | 0.836                | 0.922              | 0.775                 |



<h5>Pretrained (medium) model results on video:</h5>

<https://github.com/Masterx3/Back-Machine-Detector/assets/83991104/efc1ff09-4964-42e5-bb87-f7638a85f25a>

+ For nano model validation metrics, check [[1]](./runs/detect/train2/).

+ For medium pre-trained model validation metrics, check [[2]](./runs/detect/train/).

### Contact Me

[Linkedin](https://www.linkedin.com/in/dahshory/)

### Acknowledgements

Thanks to [Samir Gouda](github.com/SamirGouda) & [Omar Eldahshoury](github.com/omareldahshoury) for thier support.

### License

This project uses code from the YOLOv8 library by Ultralytics, which is licensed under the AGPL-3.0 License.

+ **Repository:** [Ultralytics YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
+ **License:** AGPL-3.0 License
+ **License Text:** See [LICENSE](LICENSE) for the full text.

### Citation

This project utilizes or is inspired by the RT-DETR and RTDETRv2 models. If you use this project in your work, please consider citing the original papers:

```
@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection},
      author={Yian Zhao and Wenyu Lv and Shangliang Xu and Jinman Wei and Guanzhong Wang and Qingqing Dang and Yi Liu and Jie Chen},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
      title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer}, 
      author={Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
      year={2024},
      eprint={2407.17140},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.17140}, 
}

@software{Jocher_Ultralytics_YOLO_2023,
author = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
license = {AGPL-3.0},
month = jan,
title = {{Ultralytics YOLO}},
url = {https://github.com/ultralytics/ultralytics},
version = {8.0.0},
year = {2023}
}
```
