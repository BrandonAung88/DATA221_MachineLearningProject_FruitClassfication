import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder


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

# Flatten images for Decision Tree because decision trees expect a 1d vector as input thus we flatten each image into
# a 1 dimensional feature vector this converts the 2D image into a format suitable for the model
# although it removes spatial relationships(how pixels are arranged relative to each other) between pixels which makes it so
# the program no longer knows which pixels are “next to each other” in the original images. This is a reason why
# decision trees often perform poorly on raw images compared to CNNs.
X_train_flat = X_train.reshape(len(X_train), -1)#len(X_train) is the number of images,-1 is a special numpy shortcut to auto find the dimension
X_val_flat = X_val.reshape(len(X_val), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

#Since the images are(100, 100, 3) → total pixels = 100 * 100 * 3 = 30,000,
# So after flattening, each image becomes a 1D vector of length 30,000.

# Encode labels which converts categorical labels (strings) into numbers because DecisionTreeClassifier cannot work with string labels directly
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)