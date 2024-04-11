# TFX Pipeline for Regression Model Using DNN

## Overview

This project implements a TFX (TensorFlow Extended) pipeline for training a regression model using a DNN (Deep Neural Network). TFX is an end-to-end platform for deploying production machine learning pipelines. This README provides an overview of the project, including setup instructions, pipeline components, and usage guidelines.

## Objective

The objective of this project is to develop a robust machine learning pipeline for training and deploying a regression model using TensorFlow and TFX. The regression model will be based on a DNN architecture, which is suitable for handling complex non-linear relationships in the data.

## Dataset

The dataset used in this project is the Pima Indians Diabetes Database, available on [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). This dataset contains diagnostic measurements for diabetes patients, including glucose level, blood pressure, skin thickness, and other features.

## Components

The TFX pipeline consists of the following components:

1. **ExampleGen**: This component ingests and splits input data into training and evaluation datasets.

2. **StatisticsGen**: This component computes statistics for the datasets, which are used for data analysis and schema generation.

3. **SchemaGen**: This component generates a schema based on the computed statistics, which defines the expected format of the input data.

4. **ExampleValidator**: This component validates the input data against the generated schema to detect anomalies and inconsistencies.

5. **Transform**: This component performs feature engineering and preprocessing on the input data, preparing it for training.

6. **Trainer**: This component trains the regression model using TensorFlow and Keras. It uses a DNN architecture to learn patterns and relationships in the data.

7. **Evaluator**: This component evaluates the trained model's performance using the evaluation dataset, computing metrics such as RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).

8. **Pusher**: This component deploys the trained model to a serving infrastructure, making it available for inference.