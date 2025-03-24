# ImageGenie: Intelligent Image Classification App

ImageGenie is a robust, Flask-based image classification application that leverages state-of-the-art pre-trained CNN and transformer models to predict the class of an uploaded image. With a user-friendly interface and scalable, containerized deployment, ImageGenie brings advanced image classification capabilities to your fingertips.

---

## Overview

- **Intelligent Classification:** Fine-tuned CNN and transformer-based models deliver high accuracy.
- **Flask-Based Application:** Provides a seamless interface for image uploads and real-time predictions.
- **Containerized Deployment:** Dockerized for consistent deployment across various environments.
- **Optimized Performance:** Supports multi-GPU training and inference for accelerated processing.
- **Comprehensive Data Processing:** Incorporates image resizing, normalization with ImageNet statistics, and data augmentation.

For a detailed project report, please see the [Project Report Link](https://www.overleaf.com/read/rrkdwfzrgxjz#a35998).

---

## Live Deployment

Experience ImageGenie in action at:  
[https://classification-app-r53j.onrender.com](https://classification-app-r53j.onrender.com)

---

## Docker Image

Run ImageGenie in a containerized environment using our Docker image:  
[https://hub.docker.com/r/hiikartik/classification_app](https://hub.docker.com/r/hiikartik/classification_app)

---

## Quick Start Guide

### 1. Clone the Repository

Open a terminal and run:
```bash
git clone https://github.com/hikartik/Classification_app.git
```

### 2. Set Up Your Environment

Create a virtual environment (optional but recommended):

#### On Windows (Command Prompt):
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Upgrade pip and install the required packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Running the Application

Start the Flask application:
```bash
python app.py
```

Then, open your browser and navigate to [http://localhost:5000](http://localhost:5000).  
*(If running on a VM, replace `localhost` with your VM’s public IP address.)*

---

## Project Report

Below is the comprehensive project report detailing the approach, methodology, and evaluation:

### Image Classification Using CNN and Transformer-based Models: A Comprehensive Report

**Author:** Kartik Kumar  
**Date:** March 24, 2025

#### Abstract
This report presents an end-to-end solution for image classification on the Intel Image Classification dataset. The dataset was chosen for its colorful and highly detailed images, offering a realistic challenge compared to simpler benchmarks. Due to limited training data, the test set was also used as the validation set. Various CNN and transformer-based models were fine-tuned using ImageNet pre-trained weights and optimized with multi-GPU support. The deployment of the solution is accomplished via a Flask API, which is containerized using Docker. This document details data preprocessing, model selection and optimization, and provides evaluation results for each model.

#### 1. Introduction
The Intel Image Classification dataset comprises approximately 25,000 natural scene images (150×150 pixels) distributed among six classes: buildings, forest, glacier, mountain, sea, and street. This dataset was selected because its images are more colorful and rich in pixel information compared to standard datasets like CIFAR-10 or MNIST. Such complexity makes it an excellent benchmark for modern deep learning architectures. Due to the limited training data available, the test set was also used as the validation set.

#### 2. Data Preprocessing and Feature Engineering
To prepare the images for training and evaluation, the following steps were performed:
- **Resizing:** All images were resized to 224×224 pixels to match the input size expected by the models.
- **Normalization:** Pixel values were normalized using the ImageNet mean and standard deviation.
- **Data Augmentation:** For training, random horizontal flips and rotations (up to 10 degrees) were applied to introduce variability and prevent overfitting.

#### 3. Model Selection and Optimization Approach

**Selected Models and Rationale:**
- **ResNet-18:** A classic CNN model that serves as a strong baseline. Its simplicity and proven performance make it a reliable reference.
- **MobileNet-V2:** A lightweight CNN designed for mobile applications. It demonstrates efficient performance with fewer parameters.
- **EfficientNet-B0:** Known for its efficient scaling of depth, width, and resolution, it provides high accuracy while keeping computational costs low.
- **ConvNeXt:** A modern CNN that integrates design elements from transformers. Its performance often surpasses that of traditional CNNs.
- **MobileViT:** A hybrid model combining CNN and transformer architectures, offering a balance between efficiency and accuracy.
- **Swin-Tiny:** A vision transformer with a hierarchical design using shifted windows. It provides robust performance in a compact form.
- **ViT:** The pure Vision Transformer model, which uses self-attention to capture global dependencies.

**Optimization Strategy:**
- **Transfer Learning:** All models were initialized with ImageNet pre-trained weights, and their classifier heads were replaced to output 6 classes.
- **Learning Rate Scheduling:** A StepLR scheduler was used to decay the learning rate during training.
- **Multi-GPU Support:** Training was accelerated by leveraging multiple GPUs.

#### 4. Deployment Strategy and API Usage
The final solution is deployed as a Flask API. Key deployment details include:
- **Web Interface:** Built using HTML and CSS, allowing users to upload an image and select a model for prediction.
- **Model Loading:** The API loads the corresponding saved model weights from the model/ directory and returns the predicted class.
- **Containerization:** The entire application is containerized using Docker for consistent deployment across different environments and virtual machines.

**Repository and Docker Image:**
- **Deployment Link:** [https://classification-app-r53j.onrender.com](https://classification-app-r53j.onrender.com)
- **GitHub Repository:** [https://github.com/hikartik/Classification_app.git](https://github.com/hikartik/Classification_app.git)
- **Docker Image:** [https://hub.docker.com/r/hiikartik/classification_app](https://hub.docker.com/r/hiikartik/classification_app)

#### 5. Evaluation Results
The following results were obtained on a test/validation set of 3000 images for each model:

| Model           | Accuracy | Macro F1 Score | Log Loss |
|-----------------|----------|----------------|----------|
| ConvNeXt        | 0.9503   | 0.9513         | 0.2054   |
| MobileViT       | 0.9433   | 0.9440         | 0.1785   |
| Swin-Tiny       | 0.9443   | 0.9453         | 0.2131   |
| ViT             | 0.9330   | 0.9341         | 0.3032   |
| ResNet-18       | 0.9433   | 0.9444         | 0.2003   |
| MobileNet-V2    | 0.9417   | 0.9427         | 0.1896   |
| EfficientNet-B0 | 0.9450   | 0.9461         | 0.1788   |

*Table 1: Summary of Evaluation Metrics on the Test/Validation Set.*

**Confusion Matrices and ROC Analysis:**  
Each model produced a confusion matrix that indicates high accuracy across classes. ROC AUC values for all classes were nearly 1.00, indicating excellent separability.

#### 6. Conclusion
The Intel Image Classification dataset was selected due to its colorful and highly detailed images, offering a realistic and challenging environment for image classification. A range of models—from classic CNNs (ResNet-18, MobileNet-V2, EfficientNet-B0) to modern architectures (ConvNeXt, MobileViT, Swin-Tiny, and ViT)—were evaluated. Modern architectures such as ConvNeXt and transformer-based models demonstrated state-of-the-art performance and robustness. The optimization strategy, including multi-GPU support and transfer learning, contributed significantly to the high accuracy achieved by the models. The solution is deployed as a Flask API and containerized with Docker, ensuring reproducible and scalable deployment.

For further details and updates, please refer to the [full project report](https://www.overleaf.com/read/rrkdwfzrgxjz#a35998).

---

## Technologies Used

- **Backend:** Flask, Python
- **Machine Learning:** Pre-trained CNN and Transformer Models
- **Containerization:** Docker
- **Hardware Acceleration:** Multi-GPU Support

---

## Contributions

Contributions are welcome! Please fork the repository and submit pull requests for any improvements or additional features.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or feedback, please open an issue in the repository or contact the project maintainer.
