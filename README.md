NLP Project

This project analyzes a collection of customer complaints using natural language processing (NLP) techniques. The goal of the project is to extract the most prevalent topics discussed in the complaints and gain insights into the customer feedback.

Data
The customer complaints data is obtained from the Consumer Complaint Database maintained by the Consumer Financial Protection Bureau (CFPB). The data is available as a CSV file containing over 1.7 million customer complaints spanning the years 2011-2021. For this project, a subset of 100,000 randomly selected complaints is used.

Workflow

The project consists of several phases:

Conception phase
Development phase
Finalization phase
Conception phase
The first step is to create a written concept to describe everything that belongs to the data analysis workflow. The following steps are taken in this phase:

Choose an appropriate sample data source.
Describe how to preprocess the data to achieve "clean texts".
Describe at least two approaches to convert the data into numeric vectors.
Describe at least two techniques to extract the most prevalent topics.
List the Python libraries that will be used.
Development phase
In this phase, the actual data analysis is conducted based on the concept from the conception phase. The following steps are taken in this phase:

Install all required dependencies using a Conda environment.
Load the sample data obtained in the conception phase into Python.
Preprocess the data to achieve "clean text".
Vectorize the text using Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF).
Extract the most prevalent topics using Non-negative Matrix Factorization (NMF) and Latent Dirichlet Allocation (LDA) techniques.
Discuss the results.
Finalization phase
In this phase, the final report is prepared based on the results from the development phase. The following steps are taken in this phase:

Prepare the final report.
Review and refine the report.
Submit the final report.
Code
The project code is organized as follows:

01_eda.ipynb: Exploratory data analysis of the customer complaints data.
02_preprocessing.ipynb: Text preprocessing of the customer complaints data.
03_vectorization.ipynb: Vectorization of the customer complaints data using BoW and TF-IDF.
04_topic_extraction.ipynb: Topic extraction of the customer complaints data using NMF and LDA.
preprocessing.py: Script containing text preprocessing functions.
vectorization.py: Script containing text vectorization functions.
topic_extraction.py: Script containing topic extraction functions.
environment.yml: Conda environment specification file.
data/: Directory containing the input and output data files.
Dependencies
The project dependencies are specified in the environment.yml file. To install the dependencies, create a Conda environment using the environment.yml file:

conda env create -f environment.yml
conda activate nlp-project

Usage
To run the project code, first activate the Conda environment:
conda activate nlp-project

Then run the Jupyter notebooks in order:

jupyter nbconvert --execute --inplace 01_eda.ipynb
jupyter nbconvert --execute --inplace 02_preprocessing.ipynb
jupyter nbconvert --execute --inplace 03_vectorization.ipynb
jupyter nbconvert --execute --inplace 04_topic


