import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

class Diagnosis:
    def __init__(self, model_path, weights_path):
        self.model = self.initialize_model(model_path, weights_path)

        self.root = tk.Tk()
        self.root.title("Breast Cancer Diagnosis")
        self.root.geometry("400x200")

        self.select_image_button = tk.Button(self.root, text="Select Image", command=self.select_image)
        self.select_image_button.pack(pady=20)

        self.root.mainloop()

    def initialize_model(self, model_path, weights_path):
        model = tf.keras.models.load_model(model_path)

        initial_learning_rate = 0.00001

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            metrics=["accuracy"],
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        )

        model.load_weights(weights_path)
        return model

    def sharpen_image(self, image):
        return cv2.addWeighted(image, 1.5, cv2.GaussianBlur(image, (0, 0), 3), -0.5, 0)

    def classify_image(self, image_path):
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (224, 224))
        processed_image = self.sharpen_image(resized_image)
        normalized_image = processed_image / 255.0
        reshaped_image = normalized_image.reshape(-1, 224, 224, 3)

        predictions = self.model.predict(reshaped_image)[0]

        labels = [
            'Benign with density 1',
            'Malignant with density 1',
            'Benign with density 2',
            'Malignant with density 2',
            'Benign with density 3',
            'Malignant with density 3',
            'Benign with density 4',
            'Malignant with density 4'
        ]

        prediction_results = []
        for i in range(len(labels)):
            prediction_results.append((labels[i], float(predictions[i]) * 100))

        sorted_results = sorted(prediction_results, key=lambda x: x[1], reverse=True)

        print("Prediction Results:")
        for label, percentage in sorted_results:
            print(f"{label}: {percentage:.2f}%")

        highest_label, highest_score = sorted_results[0]
        conclusion = self.generate_conclusion(highest_label)

        self.show_diagnosis(conclusion)

        labels = []
        percentages = []

        for label, percentage in sorted_results:
            labels.append(label)
            percentages.append(percentage)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, percentages, color='skyblue')
        plt.xlabel("Labels")
        plt.ylabel("Percentage")
        plt.title("Prediction Results")

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}%", va='bottom')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def generate_conclusion(self, highest_label):
        density = highest_label.split(' ')[-1]
        density_descriptions = {
            '1': "almost entirely fatty tissue, making it easier to detect abnormalities.",
            '2': "scattered areas of fibroglandular density, which may complicate detection slightly.",
            '3': "heterogeneously dense, which can obscure small tumors.",
            '4': "extremely dense, posing the greatest challenge for detection."
        }

        if "Malignant" in highest_label:
            return (f"Based on the results, there is a likelihood of breast cancer with density {density}."
                    f"This density is {density_descriptions[density]} Please consult a healthcare professional for further evaluation.")
        elif "Benign" in highest_label:
            return (f"Based on the results, it appears to be benign with density {density}. "
                    f"This density is {density_descriptions[density]} However, it is still advisable to discuss these results with a healthcare provider.")
        else:
            return "The results are inconclusive. Please consult a healthcare professional for further evaluation."

    def show_diagnosis(self, conclusion):
        diagnosis_window = tk.Toplevel(self.root)
        diagnosis_window.title("Diagnosis")

        diagnosis_window.geometry("400x200")

        label = tk.Label(diagnosis_window, text=conclusion, wraplength=380, justify="center")
        label.pack(pady=20)

    def select_image(self):
        image_path = filedialog.askopenfilename()
        self.classify_image(image_path)

if __name__ == "__main__":
    app = Diagnosis("./model/model.h5", "./model/weights.h5")