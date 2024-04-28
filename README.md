# Automated Training Pipeline for Small Language Models

## Introduction
Welcome to our Automated Training Pipeline for Small Language Models project. This initiative is designed to streamline the process of training smaller, more efficient language models, reducing the computational costs and resources required compared to their larger counterparts. This project focuses on developing a robust system that enables users to easily train and utilize n-gram models for predictive text applications.

## Project Overview
The project provides a user-friendly interface through a web application, allowing users to either input text directly or upload documents for model training. Utilizing n-gram modeling techniques, this pipeline efficiently processes the input data to train predictive models capable of suggesting subsequent text based on user input. The goal is to provide a lightweight yet effective tool for language processing tasks that can be easily scaled and deployed without the significant resource allocation typically associated with larger models.

## Features
- **Model Training Interface**: Users can interact with a simple web interface to input text or upload PDF files. The text extracted from these inputs serves as the training data for the n-gram model.
- **Dynamic Predictions**: Post-training, users can test the model by entering a sequence of words. The system will then predict the next set of words based on the learned n-gram probabilities.
- **Downloadable Model**: Once trained, the model can be downloaded as a `.pkl` file, allowing users to deploy the trained model independently in their own environments.
- **Lottie Animations**: To enhance user experience, the interface includes smooth and engaging animations on the home page, making the interaction not just functional but also visually appealing.
- **Cost-Effective Solution**: By focusing on n-gram models, this pipeline provides a cost-effective solution for language processing, requiring far less computational power and storage than larger deep learning models.

## Technical Details
The backend of the application is powered by Python, utilizing libraries such as NLTK for natural language processing tasks, Streamlit for creating the web application, and PyPDF2 for extracting text from PDF files. The choice of n-grams, a form of simpler and smaller language model, emphasizes efficiency and speed, catering to applications where quick responses are crucial without sacrificing too much accuracy.

## Computational Cost Benefits
One of the key advantages of this project is the significant reduction in computational costs. Large language models are known for their steep requirements in terms of computational resources, which include high-end GPUs and substantial amounts of electricity for training and inference. By optimizing the pipeline for n-gram models, which are substantially less resource-intensive, our project makes it feasible for smaller organizations or individual developers to implement language processing tasks. This democratizes access to natural language processing capabilities, allowing a broader range of users to incorporate sophisticated text prediction features into their applications without incurring the high costs associated with larger models.

## Use this application 
Click on the below link , you will be redirected to the web app where you will be able to use our application.

https://automatedslmtrainingpipeline-f8c4rvvxsnsgfmlgxivvh3.streamlit.app/


## Getting Started
To get started with this project, you will need to have Python installed on your machine, as well as the necessary Python libraries.

### Cloning the Repository
First, you need to clone the repository to your local machine. Open your terminal and run the following command:

```bash
git clone https://github.com/amith-2001/automated_slm_training_pipeline.git
cd automated_slm_training_pipeline
```
### Installing Dependencies
Before running the application, you need to install the required libraries:

```bash
 pip install -r requirements.txt
```
### Running the Application
To run the application, execute the following command in the terminal:

```streamlit run app.py --server.baseUrlPath /.streamlit/config.toml```
