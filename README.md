# 🔢 Handwritten Digit Recognizer (MNIST CNN)

A deep learning project in Python that trains a Convolutional Neural Network (CNN) on the classic MNIST dataset to achieve **$\ge 99\%$ test accuracy**, paired with an interactive **Streamlit web application** featuring a real-time drawing canvas and image uploader.

---

## 🌟 Key Features

- **High-Accuracy CNN**: Multi-stage Convolutional Neural Network with Batch Normalization, Dropout, and Adam optimizer reaching **$\ge 99\%$ accuracy** on the 10,000-image MNIST test set.
- **Robust Data Augmentation**: Random rotations, translations, and zooms during training prevent overfitting and improve generalization on freehand user drawings.
- **Interactive Streamlit Web App**:
  - **Drawing Canvas**: Freehand digit drawing with adjustable brush thickness using `streamlit-drawable-canvas`.
  - **File Uploader**: Upload scanned or photographed digits (`.png`, `.jpg`, `.jpeg`).
  - **MNIST Standard Preprocessing**: Automatic brightness/contrast inversion, bounding box extraction, aspect-ratio preservation, and **center-of-mass alignment** to faithfully match MNIST distribution.
  - **Live Probabilities**: Visual bar chart displaying prediction confidence across all 10 digit classes (0–9).
  - **Preprocessed Tensor Inspection**: Side-by-side view showing the raw drawing vs. the processed $28 \times 28$ matrix fed to the network.
- **Comprehensive Evaluation**: Automated generation of loss/accuracy learning curves and a confusion matrix heatmap.

---

## 📁 Project Structure

```text
digit-recognizer/
├── model.py                # CNN model architecture definition & compiler
├── train.py                # Training pipeline, augmentation, evaluation & plot export
├── app.py                  # Streamlit web application with drawing canvas
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation & instructions
├── model.keras             # Trained Keras model weights & architecture
├── model.h5                # Legacy HDF5 model format
├── training_curves.png     # Train vs. Validation Loss and Accuracy plot
└── confusion_matrix.png    # 10x10 MNIST test set confusion matrix heatmap
```

---

## ⚙️ Installation & Environment Setup

### 1. Clone or Navigate to the Workspace
```powershell
cd c:\Users\imshu\digit-recognizer
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🚀 How to Train the Model

To train the CNN model from scratch and evaluate its performance on the MNIST test set:

```powershell
python train.py
```

### What happens during training:
1. Downloads and normalizes the MNIST dataset ($60,000$ training / $10,000$ test images).
2. Sets up real-time data augmentation (rotation $\pm 8\%$, translation $\pm 8\%$, zoom $\pm 8\%$).
3. Trains the CNN for up to 12–15 epochs using Adam optimizer (`lr=0.001`), `ReduceLROnPlateau`, and `ModelCheckpoint`.
4. Saves the best model checkpoint to `model.keras` and `model.h5`.
5. Evaluates on the unseen test set, printing test loss, accuracy, and a full per-class classification report.
6. Generates and saves:
   - `training_curves.png`: Training vs. Validation Loss & Accuracy curves.
   - `confusion_matrix.png`: Heatmap of predictions across digits 0–9.

---

## 🌐 How to Launch the Web App

Launch the Streamlit web dashboard:

```powershell
streamlit run app.py
```

Open your browser to:
```
http://localhost:8501
```

### Using the App:
1. **✏️ Draw on Canvas**:
   - Use your mouse to draw any digit from `0` to `9` inside the canvas.
   - Adjust the brush thickness using the sidebar slider if desired.
   - The app dynamically predicts the digit, displays the confidence percentage, and renders class probabilities.
   - Click **Reset / Clear** to clear the canvas.
2. **📁 Upload Image**:
   - Switch to the "Upload Image" mode in the sidebar.
   - Upload any handwriting sample (dark ink on light paper or white ink on dark paper).
   - The app automatically detects background brightness and inverts/scales appropriately.

---

## 🧠 Model Architecture & Methodology

| Layer | Type | Specifications | Output Shape |
|---|---|---|---|
| Input | `InputLayer` | Grayscale $28 \times 28$ image | `(28, 28, 1)` |
| Conv Block 1 | `Conv2D` $\times$ 2 + `BatchNorm` | 32 filters, $3 \times 3$, ReLU, padding='same' | `(28, 28, 32)` |
| Pool 1 | `MaxPooling2D` + `Dropout` | $2 \times 2$ pool, $25\%$ dropout | `(14, 14, 32)` |
| Conv Block 2 | `Conv2D` $\times$ 2 + `BatchNorm` | 64 filters, $3 \times 3$, ReLU, padding='same' | `(14, 14, 64)` |
| Pool 2 | `MaxPooling2D` + `Dropout` | $2 \times 2$ pool, $25\%$ dropout | `(7, 7, 64)` |
| Dense | `Dense` + `BatchNorm` + `Dropout` | 256 units, ReLU, $40\%$ dropout | `(256,)` |
| Output | `Dense` | 10 units, Softmax activation | `(10,)` |

### MNIST Center-of-Mass Preprocessing:
Raw freehand drawings vary wildly in position and scale. To match the statistical properties of the MNIST training set, `app.py` implements:
1. **Bounding Box Cropping**: Isolates the drawn stroke from empty margins.
2. **Aspect Ratio Preservation**: Rescales the digit to fit into a $20 \times 20$ box without distortion.
3. **Padding**: Places the $20 \times 20$ digit inside a $28 \times 28$ frame ($4\text{px}$ margin).
4. **Center of Mass Shift**: Aligns the center of mass $(\bar{x}, \bar{y})$ directly to $(14, 14)$ using SciPy's `ndimage.center_of_mass`.

---

## 📊 Sample Results

- **Test Accuracy**: $\ge 99.1\%$
- **Test Loss**: $\approx 0.024$
- **Confusion Matrix**: Clean diagonal dominance with minimal misclassifications between notoriously similar digits (e.g., 4 vs 9, 3 vs 5).

---

## 🛠️ Troubleshooting

- **PowerShell Execution Policy Error**:
  If `Activate.ps1` produces a script execution policy error, run:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- **Blank Canvas Output**:
  Draw larger and clearer strokes, or increase the brush width to ~20px for optimal recognition.
