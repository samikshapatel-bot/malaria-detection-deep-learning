🦠 Malaria Detection Using Deep Learning
Project Overview

Malaria is a life-threatening disease caused by parasites transmitted through mosquito bites. Early detection is important for proper treatment.

This project uses **Deep Learning and Computer Vision** to automatically detect **malaria-infected cells from microscopic blood smear images**.

A **Convolutional Neural Network (CNN)** model is trained to classify images into:

* Parasitized (infected)
* Uninfected (healthy)

The trained model is integrated into a **Flask web application**, allowing users to upload a blood cell image and receive an instant prediction.

🎯 Objectives
* Detect malaria infection using microscopic images
* Use Deep Learning for medical image classification
* Build an automated diagnostic system
* Create a web application for easy prediction

🧠 Technologies Used
* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Flask
* HTML / CSS

📂 Project Structure
Malaria-Detection-Using-Deep-Learning
│
├── dataset
│   ├── train
│   │   ├── Parasitized
│   │   └── Uninfected
│   │
│   └── validation
│       ├── Parasitized
│       └── Uninfected
│
├── static
│   └── uploads
│
├── templates
│   └── index.html
│
├── visualize.py
├── augmentation.py
├── model_building.py
├── train_model.py
├── evaluate_model.py
│
├── app.py
│
├── malaria_model.h5
│
├── requirements.txt

📊 Dataset

The dataset contains **microscopic blood smear images** divided into two categories:

* **Parasitized** → infected red blood cells
* **Uninfected** → healthy red blood cells

Images are used for training and validation of the model.

Explanation of Each Folder/File
File / Folder	                         Purpose
dataset                       	      Plant disease images

*train                            	    Training images
*valid                          	      Validation images
*static/uploads	                      Uploaded images from web app
*templates	                            HTML pages for web interface
*index.html	                          Web page for image upload
*visualize.py	                        Dataset visualization
*augmentation.py	                      Image augmentation
*model.py	                            MobileNetV2 model architecture
*train_model.py	                      Model training
*evaluate_model.py                 	  Model evaluation
*app.py	                              Flask web application
*best_plant_disease_model.h5	          Trained deep learning model
*requirements.txt	                    Python libraries needed
*README.md                          	  Project documentation

⚙️ Installation
Clone the repository:
git clone https://github.com/yourusername/malaria-detection-deep-learning.git

Move into the project folder:
cd malaria-detection-deep-learning

Install required libraries:
pip install -r requirements.txt

🏋️ Model Training

Steps used for training the model:
1️⃣ Data preprocessing
2️⃣ Image augmentation
3️⃣ Dataset splitting (training & validation)
4️⃣ Model building using CNN / MobileNetV2
5️⃣ Model training
6️⃣ Model evaluation


