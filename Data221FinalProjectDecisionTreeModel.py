import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


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

# Train model random_state is set to 42 just to make it a fixed seed so results are reproducible.
# Without random_state, every time you run the command you might get slightly different trees and therefore slightly different predictions.
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_flat, y_train_enc)

y_pred = model.predict(X_test_flat)

# print("Confusion Matrix:")#row 1:apple , row2: avocado, row 3: banana, row 4: orange
cm = confusion_matrix(y_test_enc, y_pred)
# print(cm)
print("\nClassification Report:")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

# Confusion Matrix with matplotlib
plt.imshow(cm)
plt.xticks(range(len(le.classes_)), le.classes_)
plt.yticks(range(len(le.classes_)), le.classes_)

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i][j], ha='center', va='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Misclassified images
misclassified = np.where(y_pred != y_test_enc)[0]

print("Number of misclassified images:", len(misclassified))

for i in misclassified[:5]:#first 5 images that are misclassified
    plt.imshow(X_test[i])
    plt.title(f"True: {le.inverse_transform([y_test_enc[i]])[0]}, " f"Pred: {le.inverse_transform([y_pred[i]])[0]}")
    #inverse_transform decodes (numbers → words) because le(label encoder) does the opposite
    # We need this to convert the model’s numeric predictions back into human-readable labels
    # (ex: 0 → "Apple") so we can understand and display the results.
    plt.axis('off')
    plt.show()