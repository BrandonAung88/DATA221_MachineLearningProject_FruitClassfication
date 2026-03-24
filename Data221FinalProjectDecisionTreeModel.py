import os
import numpy as np
from PIL import Image


def load_images_from_folder(folder):
    X = []
    y = []

    for label in os.listdir(folder):
        class_path = os.path.join(folder, label)

        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((100, 100))

                X.append(np.array(img))
                y.append(label)
            except:
                continue

    return np.array(X), np.array(y)


# Load datasets
X_train, y_train = load_images_from_folder("Project_Train")
X_val, y_val = load_images_from_folder("Project_Val")
X_test, y_test = load_images_from_folder("Project_Test")

# Normalize pixel values to between 0 and 1
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0