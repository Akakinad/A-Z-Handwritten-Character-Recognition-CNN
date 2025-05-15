# A–Z Handwritten Character Recognition using CNN

This project builds a **Convolutional Neural Network (CNN)** to classify images of handwritten English capital letters (A–Z) from grayscale 28x28 images. The model is trained and evaluated on the popular **"A_Z Handwritten Data"** dataset.

---

## 📁 Dataset

- Source: Kaggle – [A_Z Handwritten Data](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)
- 26 classes (A–Z), each image is 28x28 grayscale pixels.
- Format: CSV with pixel values in flattened format and labels as integers (0=A, 25=Z).
- Training size: 250,000 samples  
- Test size: 100,000 samples

---

## 🛠 Features

✅ Clean, modular, and well-commented Jupyter Notebook  
✅ CNN architecture from scratch using TensorFlow/Keras  
✅ Live training and validation tracking  
✅ Evaluation using:
- Accuracy metrics
- Classification report
- Per-class accuracy breakdown
- Confusion matrix

✅ Visuals:
- Class frequency histograms
- Sample training images
- Training/validation curves
- Correct prediction samples
- Confusion matrix heatmap

---

## 🚀 How to Run

1. Clone the repository:
   git clone https://github.com/akakinad/az-handwritten-cnn.git
   cd az-handwritten-cnn

2. Install dependencies:
    pip install -r requirements.txt

3. Launch the notebook:
    jupyter notebook Az_Handwritten_CNN.ipynb

4. Ensure the dataset is placed in:
    dataset/A_Z Handwritten Data.csv

🧠 Model Architecture
    Input: (28, 28, 1)
    Convolutional Layers: 6 Conv2D layers with ReLU
    Pooling Layers: 3 MaxPooling2D layers
    Fully Connected Layers: 2 Dense layers with ReLU (256 units)
    Output Layer: Dense(26) with Softmax activation
    Optimizer: Adam
    Loss Function: Categorical Crossentropy
    Callbacks: EarlyStopping on validation accuracy 

📊 Performance Summary
    Training Accuracy: ~99.95%
    Validation Accuracy: ~99.60%
    Test Accuracy: ~99.50%
    <sub>(Exact results may vary slightly depending on shuffling/random seed)</sub>

📈 Visual Examples
    Correct Predictions	
    Confusion Matrix

🧾 Classification Report (Excerpt)
    markdown
              precision    recall  f1-score   support

           A       0.99      1.00      1.00      3844
           B       0.99      0.99      0.99      3701
           ...
           Z       1.00      1.00      1.00      3802

    accuracy                           0.99    100000
   macro avg       0.99      0.99      0.99    100000
weighted avg       0.99      0.99      0.99    100000

📌 Future Improvements
    Add dropout or regularization
    Implement model saving/loading
    Build a web UI using Flask or Streamlit for real-time character prediction

👨‍💻 Author: Akakinad
    Feel free to reach out on LinkedIn or GitHub for questions or collaborations.

📄 License
MIT License - use, modify, and distribute freely.
