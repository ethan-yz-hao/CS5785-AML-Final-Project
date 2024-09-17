# -*- coding: utf-8 -*-
"""
@Time : 12/09/2023 1:31 PM
@Auth : Hao Yizhi
@File : SVM.py
@IDE  : PyCharm
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.feature import hog
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import roc_curve
from tqdm import tqdm
from pathlib import Path
import joblib
import logging
import time


# Improved function to load images and extract HOG features with error handling and progress bar
def load_images_and_extract_features(directory, label, image_size=(32, 32), hog_params={}, max_images=None):
    features = []
    labels = []
    files = list(Path(directory).glob('*'))  # Using pathlib for improved path handling
    if max_images:
        files = files[:max_images]
    for file in tqdm(files, desc=f"Processing {label} images", total=len(files)):
        try:
            image = imread(str(file), as_gray=True)  # as_gray depends on image type
            image_resized = resize(image, image_size, anti_aliasing=True)
            hog_feature = hog(image_resized, **hog_params)
            features.append(hog_feature)
            labels.append(label)
        except Exception as e:  # This will catch any issues during image loading
            print(f"Error processing {file}: {e}")
    return features, labels

if __name__ == '__main__':
    local_time = time.localtime()
    formatted_time = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)

    # Set up logging
    logging.basicConfig(
        filename=f'svm/{formatted_time} - SVM_rbf.txt',  # Change the extension to .txt
        level=logging.INFO,
        filemode='w',  # 'w' to overwrite the log file on each run, 'a' to append
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Including a timestamp
        datefmt='%Y-%m-%d %H:%M:%S'  # Including date format
    )

    # Parameters for controlling the amount of data used (for example purposes)
    max_train_images_per_class = None  # Use None to load all images
    max_test_images_per_class = None  # Use None to load all images

    # Parameters for HOG feature extraction
    hog_params = {
        'pixels_per_cell': (8, 8),
        'cells_per_block': (2, 2),
        'multichannel': False
    }

    # Define directories (adjust the paths according to your dataset)
    train_fake_dir = '../../Dataset/train/FAKE'
    train_real_dir = '../../Dataset/train/REAL'
    test_fake_dir = '../../Dataset/test/FAKE'
    test_real_dir = '../../Dataset/test/REAL'

    # Load and extract features for training data
    train_fake_features, train_fake_labels = load_images_and_extract_features(train_fake_dir,
                                                                              0,
                                                                              hog_params=hog_params,
                                                                              max_images=max_train_images_per_class)
    train_real_features, train_real_labels = load_images_and_extract_features(train_real_dir,
                                                                              1,
                                                                              hog_params=hog_params,
                                                                              max_images=max_train_images_per_class)

    # Combine training data
    train_features = np.array(train_fake_features + train_real_features)
    train_labels = np.array(train_fake_labels + train_real_labels)

    # Load and extract features for testing data
    test_fake_features, test_fake_labels = load_images_and_extract_features(test_fake_dir, 0, hog_params=hog_params,
                                                                            max_images=max_test_images_per_class)
    test_real_features, test_real_labels = load_images_and_extract_features(test_real_dir, 1, hog_params=hog_params,
                                                                            max_images=max_test_images_per_class)

    # Combine testing data
    test_features = np.array(test_fake_features + test_real_features)
    test_labels = np.array(test_fake_labels + test_real_labels)

    # Normalize features using a standard scaler
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    # Train SVM
    svm_classifier = SVC(kernel='rbf', probability=True)
    svm_classifier.fit(train_features_scaled, train_labels)

    # Evaluate the classifier
    test_predictions = svm_classifier.predict(test_features_scaled)
    test_scores = svm_classifier.decision_function(test_features_scaled)

    # Print evaluation metrics
    logging.info("Accuracy:", accuracy_score(test_labels, test_predictions))
    logging.info("Classification Report:", classification_report(test_labels, test_predictions))
    logging.info("Confusion Matrix:", confusion_matrix(test_labels, test_predictions))
    roc_auc = roc_auc_score(test_labels, test_scores)
    logging.info("ROC-AUC Score:", roc_auc)
    precision, recall, _ = precision_recall_curve(test_labels, test_scores)
    pr_auc = auc(recall, precision)
    logging.info("Precision-Recall AUC:", pr_auc)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"svm/{formatted_time} - SVM_rbf_ROC_curve.png")  # Save the ROC curve plot
    plt.show()
    plt.close()

    # Save the trained model for future use
    joblib.dump(svm_classifier, f'svm/{formatted_time}SVM_rbf_model.joblib')
