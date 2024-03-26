# YOLOv8 Number Plate Detection using Colab

This Colab notebook implements YOLOv8 for number plate detection. The model is trained to detect number plates in images of vehicles with high accuracy.

## Overview

This notebook provides a step-by-step guide on how to train and evaluate the YOLOv8 model for number plate detection using Google Colab.

## Requirements

- Google account
- Google Colab

## Usage

1. Open the Colab notebook: [YOLOv8_Number_Plate_Detection.ipynb](https://github.com/Apiljungthapa/YOLOv8-with-license_plateDetectection/blob/main/Training_Yolov8CustomDatasets.ipynb).

2. Follow the instructions provided in the notebook to run each code cell sequentially.

3. The notebook includes sections for data preparation, model training, evaluation, and inference.

## Dataset Preparation

1. Organize your dataset within the "Main Data" folder as follows:
   
Main Data
│
├── images
│   ├── train
│   └── val
│
└── labels
    ├── train
    └── val


Place your training images in the `images/train` folder and validation images in the `images/val` folder. Corresponding label files should be placed in the `labels/train` and `labels/val` folders, following the YOLO format.

2. Use the Roboflow website (https://roboflow.com/) to create and annotate your dataset of images containing vehicles with annotated number plates.

3. Export the dataset in YOLO format.

4. You can download the dataset manually as a zip folder or use the Roboflow API for automated download.

 - For manual download, visit the dataset page on Roboflow and click the "Download" button.
 
 - For automated download using the Roboflow API, you can use the provided API key. The API documentation can be found [here](https://app.roboflow.com/).

## Model Training

1. Configure the training parameters in the notebook, including batch size, number of epochs, and learning rate.

2. Run the training cells to start training the YOLOv8 model.

3. During training, all outputs including images of all batches, confusion matrix, and other metrics will be saved in the folder path `runs/detect/train`.

## Model Evaluation

1. After training, evaluate the trained model using the evaluation cells provided in the notebook.

2. Compute performance metrics such as precision, recall, and mAP (mean Average Precision) on a test set.

3. Evaluation outputs, including the confusion matrix and other metrics, will be saved in the folder path `runs/detect/train`.

## Inference

1. Use the trained model to perform inference on new images or videos containing vehicles.

2. The notebook includes examples of how to perform inference using both single images and video files.

3. Inference results and detected images will be saved in the folder path `runs/detect/predict`.

## Results

1. The notebook provides visualizations of the model's performance during training and evaluation.

2. Sample detection results are included to showcase the model's accuracy in detecting number plates.

## Acknowledgments

- This notebook is based on the YOLOv8 implementation by [https://github.com/Apiljungthapa/YOLOv8-with-license_plateDetectection/blob/main/Training_Yolov8CustomDatasets.ipynb].

- Thanks to the authors of the dataset used for training and evaluation.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Apiljungthapa/YOLOv8-with-license_plateDetectection/blob/main/LICENSE) file for details.
