Project Overview
This project focuses on training and evaluating a ResNet model for facial expression recognition using the DISFA dataset. The codebase includes scripts for training, testing, and analyzing the model's performance.

-------
File Descriptions
training_analysis.ipynb
This Jupyter notebook is used for analyzing the training results of the ResNet model, including parsing log files, visualizing training and test loss over epochs, and displaying loss results in a tabular format.

visualization_demo.ipynb
This Jupyter notebook demonstrates how to load a trained ResNet model and visualize its predictions on test images, including loading the model architecture and processing images for evaluation.

test_all.py
This script is designed to evaluate the performance of the trained ResNet model across multiple folds of the dataset, calculating and logging the average loss for each model.

my_scripts/trained_resnet_with_fold_X/loss.log
These log files contain the training and test loss information for each fold of the ResNet model, where X represents the fold number. They are used for tracking the model's performance during training.

best_models.txt
This text file lists the best-performing models for each fold, along with their corresponding loss values, allowing for easy identification of the most effective models.

train_resnet.py
This script is responsible for training the ResNet model on the DISFA dataset, including data loading, model initialization, and logging the training loss to a file.

loss_results.txt
This file summarizes the total loss for different models across various folds, providing a quick reference for evaluating model performance.

test.py
This script is used for testing a specific trained ResNet model on the test dataset, calculating the total loss and providing both scaled and unscaled loss outputs.
-------

Usage
To run the training or testing scripts, ensure that the required dependencies are installed and the DISFA dataset is properly set up. Use the command line to execute the scripts with appropriate arguments.