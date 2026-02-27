# ImageFocusDetection

***

## Description (temporary)

This project studies image focus detection by reformulating it as a binary classification problem at the patch level.
Instead of directly predicting the focus region of an entire image, each image is divided into multiple non-overlapping
patches,
and each patch is classified as either sharp or blurred. Based on the patch-level predictions, the focus distribution of
the original image
can be reconstructed. The project compares traditional machine learning methods based on handcrafted features with
lightweight deep learning
models, and investigates how different patch sizes affect classification performance and focus localization accuracy.

## Project plan

1. Dataset Preparation  
   The first step is to prepare the dataset using both publicly available image datasets and manually collected images.
   Clear images and blurred images are obtained either from existing datasets or by applying synthetic blur to sharp
   images.
   The dataset is then organized and split into training, validation, and test sets.

2. Image Patch Generation  
   Each image is divided into multiple fixed-size patches. Different patch sizes are considered in order to study their
   influence on
   classification performance and focus localization accuracy. Each patch is assigned a binary label indicating whether
   it is sharp or blurred.

3. Traditional Machine Learning Classifiers  
   Handcrafted sharpness-related features are extracted from each image patch. Several classical machine learning
   classifiers are implemented
   and trained, such as Support Vector Machines and Random Forests, to serve as baseline methods for the binary
   classification task.

4. Lightweight Convolutional Neural Network  
   A simple and lightweight convolutional neural network is designed and trained to perform patch-level sharpness
   classification.
   The network aims to automatically learn discriminative features while maintaining low computational complexity.

5. Performance Evaluation and Comparison  
   Different machine learning and deep learning methods are evaluated and compared on the binary classification task
   using standard
   metrics such as accuracy, precision, recall, and F1-score. The performance differences between traditional machine
   learning approaches
   and lightweight deep learning models are analyzed and discussed.