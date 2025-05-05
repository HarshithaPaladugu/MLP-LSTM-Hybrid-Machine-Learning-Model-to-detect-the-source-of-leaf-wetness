# MLP-LSTM-Machine-Learning-Model-to-detect-the-source-of-leaf-wetness
Here's a well-structured and detailed **README.md** content for your GitHub repository based on the methodology you provided. It explains the purpose, data, model, techniques used, and visualization clearly for readers or collaborators:

---

# 🌿 Leaf Wetness Source Detection Using Hybrid MLP-LSTM Model

This repository contains the complete implementation for detecting the source of **leaf wetness** in **chili plants** using data collected from **leaf wetness sensors**. The project combines **Multilayer Perceptron (MLP)** and **Long Short-Term Memory (LSTM)** in a hybrid deep learning architecture to classify the source of wetness as **dew**, **rainfall**, or **irrigation**.

---

## 📊 Dataset Overview

* **Total Records Collected**: 6919
* **Duration**: 5 months (May to September)
* **Sampling Interval**: Every 20 minutes
* **Collected Using**: Leaf moisture sensor deployed in chili fields
* **Columns Used**:

  * `leaf_temperature`
  * `leaf_moisture`
  * `class` *(derived column for classification target)*

Other columns like `Date`, `Time`, `Device Name`, `bat`, `rssi`, `snr`, and `spreading_factor` were discarded as they are not relevant for classification.

---

## 🧹 Data Preprocessing

The raw data was cleaned and preprocessed using the following steps:

1. **Outlier Handling**:
   Applied **IQR method** to detect and remove outliers from the `leaf_moisture` column.

2. **NULL Values**:
   No missing values were found in the cleaned dataset. Mode, mean, or median imputation would be used otherwise, based on column data type.

3. **Skewness Check**:
   Verified if columns were normally distributed using skewness metric and normalized if required.

4. **Class Labeling Rules**:

   * `dew`: if `leaf_temperature < 25` and `60 <= leaf_moisture <= 75`
   * `rainfall`: if `20 <= leaf_temperature <= 30` and `leaf_moisture > 80`
   * `irrigation`: if `25 <= leaf_temperature <= 35` and `50 <= leaf_moisture <= 70`

After preprocessing, the dataset size reduced to **3304 records**.

---

## ⚖️ Dataset Balancing - SMOTE

To address class imbalance, **Custom SMOTE** was applied to balance the dataset across the three classes:

| Class      | Before SMOTE | After SMOTE |
| ---------- | ------------ | ----------- |
| Dew        | 1888         | 1146        |
| Irrigation | 1171         | 1267        |
| Rainfall   | 245          | 1307        |

This technique avoided overfitting by generating synthetic samples in user-specified quantities.

---

## 🧠 Model Architecture - MLP + LSTM (Hybrid)

A hybrid model combining **MLP** and **LSTM** was developed to capture both feature-based and sequential dependencies in the time-series data.

### Key Features:

* **MLP layers**: Feature learning from input data
* **LSTM layer**: Sequential pattern detection from time-series inputs
* **Dropout**: Regularization to prevent overfitting
* **L2 Regularization**: Applied in LSTM layer
* **Activation Functions**:

  * `ReLU` in hidden layers
  * `Softmax` in output layer for multiclass classification

### Input Features:

* `leaf_temperature`
* `leaf_moisture`

### Output:

* Probabilities for 3 classes: `dew`, `rainfall`, `irrigation`

### Data Split:

* **Training**: 80%
* **Validation**: 10%
* **Testing**: 10%

Data was saved and reused in `.pkl` files.

---

## 📈 Model Evaluation & Visualization

Model performance was evaluated using **accuracy** and **loss** metrics over training and validation sets.

### Results:

* **Hybrid MLP-LSTM Accuracy**: 99.73%
* **Standalone MLP Accuracy**: 92.86%

### Visualizations:

* **Accuracy vs Epochs**: `Fig. 13`
* **Loss vs Epochs**: `Fig. 14`

These plots demonstrate improved generalization and reduced overfitting in the hybrid model compared to standalone models.

---

## 📁 Repository Structure

```bash
├── data/
│   ├── raw_data.csv
│   ├── cleaned_data.csv
│   └── train_test_val.pkl
├── models/
│   ├── mlp_lstm_model.h5
│   └── model_architecture.png
├── notebooks/
│   └── preprocessing_and_model.ipynb
├── utils/
│   └── smote_helper.py
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/leaf-wetness-mlp-lstm.git
   cd leaf-wetness-mlp-lstm
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:

   ```bash
   jupyter notebook notebooks/preprocessing_and_model.ipynb
   ```

---

## 📚 Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib & Seaborn
* SMOTE (Imbalanced-learn)

---

## 👩‍🔬 Research Contribution

This repository is part of a research project that focuses on using **IoT sensor data** and **hybrid deep learning models** to improve detection of agricultural micro-climatic conditions. By combining MLP with LSTM, the model achieves significantly higher accuracy and robustness in time-series prediction tasks.

---

## 📩 Contact

For any queries or collaborations, reach out at:

**Harshitha Paladugu**
📧 [paladugu.harshitha2000@gmail.com](mailto:paladugu.harshitha2000@gmail.com)
📱 +91-9059017283
