import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def load_images_from_folder(folder_path):
    image_data = []
    image_labels = []

    for label in os.listdir(folder_path):
        class_path = os.path.join(folder_path, label)

        if not os.path.isdir(class_path):
            continue

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            try:
                image = Image.open(image_path).convert('RGB')
                image = image.resize((100, 100))

                image_data.append(np.array(image))
                image_labels.append(label)
            except:
                continue

    return np.array(image_data), np.array(image_labels)


# Load datasets
training_images, training_labels = load_images_from_folder("Project_Train")
validation_images, validation_labels = load_images_from_folder("Project_Val")
testing_images, testing_labels = load_images_from_folder("Project_Test")

# Normalize pixel values to between 0 and 1
training_images = training_images / 255.0
validation_images = validation_images / 255.0
testing_images = testing_images / 255.0

# Flatten images for Decision Tree because decision trees expect a 1d vector as input thus we flatten each image into
# a 1 dimensional feature vector this converts the 2D image into a format suitable for the model
# although it removes spatial relationships(how pixels are arranged relative to each other) between pixels which makes it so
# the program no longer knows which pixels are “next to each other” in the original images. This is a reason why
# decision trees often perform poorly on raw images compared to CNNs.
training_features = training_images.reshape(len(training_images), -1)#len(X_train) is the number of images,-1 is a special numpy shortcut to auto find the dimension
validation_features = validation_images.reshape(len(validation_images), -1)
testing_features = testing_images.reshape(len(testing_images), -1)

#Since the images are(100, 100, 3) → total pixels = 100 * 100 * 3 = 30,000,
# So after flattening, each image becomes a 1D vector of length 30,000.

# Encode labels which converts categorical labels (strings) into numbers because DecisionTreeClassifier cannot work with string labels directly
label_encoder = LabelEncoder()
training_labels_encoded = label_encoder.fit_transform(training_labels)
validation_labels_encoded = label_encoder.transform(validation_labels)
testing_labels_encoded = label_encoder.transform(testing_labels)

# Train model random_state is set to 42 just to make it a fixed seed so results are reproducible.
# Without random_state, every time you run the command you might get slightly different trees and therefore slightly different predictions.
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
decision_tree_classifier.fit(training_features, training_labels_encoded)

predicted_testing_labels = decision_tree_classifier.predict(testing_features)

# print("Confusion Matrix:")#row 1:apple , row2: avocado, row 3: banana, row 4: orange
confusion_matrix_result = confusion_matrix(testing_labels_encoded, predicted_testing_labels)
# print(cm)
print("\nClassification Report:")
print(classification_report(testing_labels_encoded, predicted_testing_labels, target_names=label_encoder.classes_))

# Confusion Matrix with matplotlib
plt.imshow(confusion_matrix_result)
plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_)
plt.yticks(range(len(label_encoder.classes_)), label_encoder.classes_)

for row_index in range(len(confusion_matrix_result)):
    for column_index in range(len(confusion_matrix_result)):
        plt.text(column_index, row_index, confusion_matrix_result[row_index][column_index], ha='center', va='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Misclassified images
misclassified_indices = np.where(predicted_testing_labels != testing_labels_encoded)[0]

print("Number of misclassified images:", len(misclassified_indices))

for index in misclassified_indices[:5]:#first 5 images that are misclassified
    plt.imshow(testing_images[index])
    plt.title(
        f"True: {label_encoder.inverse_transform([testing_labels_encoded[index]])[0]}, "
        f"Pred: {label_encoder.inverse_transform([predicted_testing_labels[index]])[0]}"
    )
    #inverse_transform decodes (numbers → words) because le(label encoder) does the opposite
    # We need this to convert the model’s numeric predictions back into human-readable labels
    # (ex: 0 → "Apple") so we can understand and display the results.
    plt.axis('off')
    plt.show()