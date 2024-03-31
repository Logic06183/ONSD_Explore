# ONSD_Explore

# Ultrasound ONSD Measurement Prediction

This repository contains a collection of models and notebooks aimed at predicting the Optic Nerve Sheath Diameter (ONSD) from ultrasound images. Our approach utilizes various machine learning and deep learning techniques, including traditional models, gradient boosting with XGBoost, and transfer learning with deep neural networks such as ResNet50.

## Project Structure

- `documents/`: Contains ultrasound images and related documents.
- `Meta_pic.xlsx`: Meta information about the ultrasound images.
- `ONSD_explore.ipynb`: A Jupyter notebook for exploratory data analysis.
- `Other_models_ONSD.ipynb`: A notebook with implementations of other models.
- `Transfer_Learning.ipynb`: A notebook detailing the transfer learning approach with ResNet50.
- `image_labels.csv`: A CSV file linking images to their respective ONSD measurements.
- `LICENSE`: The license file for the project.

## Installation

Before running the notebooks, ensure you have the necessary libraries installed. It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
