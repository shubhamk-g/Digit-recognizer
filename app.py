"""
app.py - Streamlit Web Application for Handwritten Digit Recognition.
Features:
  - Interactive drawing canvas (streamlit-drawable-canvas)
  - Image file upload (PNG, JPG, JPEG)
  - Standard MNIST preprocessing (crop to bounding box, preserve aspect ratio, center of mass alignment, 28x28 normalization)
  - Real-time prediction, confidence score, and 10-class probability distribution chart
  - Inspection preview of the preprocessed 28x28 image
"""

import os
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
import streamlit as st
import pandas as pd
import tensorflow as tf
def get_st_canvas():
    """Lazily imports st_canvas within the active Streamlit runtime context."""
    from streamlit_drawable_canvas import st_canvas
    return st_canvas

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }

    .card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }

    .pred-card {
        text-align: center;
        padding: 20px;
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.15);
    }

    .pred-digit {
        font-family: 'JetBrains Mono', monospace;
        font-size: 5rem;
        font-weight: 800;
        color: #38bdf8;
        line-height: 1;
        margin: 10px 0;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.5);
    }

    .confidence-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1rem;
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    .top-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(51, 65, 85, 0.6);
        padding: 4px 12px;
        border-radius: 8px;
        margin: 4px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_trained_model():
    """Loads the pre-trained MNIST CNN model."""
    candidates = ["model.keras", "model.h5"]
    for path in candidates:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                return model, path
            except Exception as e:
                st.warning(f"Failed to load {path}: {e}")
    return None, None


def center_digit_by_mass(image_28x28: np.ndarray) -> np.ndarray:
    """
    Centers the digit image using center of mass, matching MNIST's original standard.
    """
    cy, cx = ndimage.center_of_mass(image_28x28)
    if np.isnan(cx) or np.isnan(cy):
        return image_28x28

    shift_x = np.round(14.0 - cx).astype(int)
    shift_y = np.round(14.0 - cy).astype(int)

    # Constrain shift so digit doesn't get pushed out of frame
    shift_x = max(min(shift_x, 6), -6)
    shift_y = max(min(shift_y, 6), -6)

    centered = ndimage.shift(image_28x28, shift=(shift_y, shift_x), mode="constant", cval=0.0)
    return centered


def preprocess_image(image_input) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard MNIST Preprocessing Pipeline:
      1. Convert to grayscale.
      2. Detect if inversion is needed (digit should be white, background black).
      3. Crop around non-zero pixels (bounding box).
      4. Fit bounding box into 20x20 box preserving aspect ratio.
      5. Pad into 28x28 image.
      6. Align center of mass.
      7. Normalize to [0.0, 1.0].

    Returns:
        tuple (processed_28x28_for_display, model_input_tensor_1_28_28_1)
    """
    if isinstance(image_input, np.ndarray):
        # Canvas RGBA array
        if image_input.shape[-1] == 4:
            # Alpha channel mask or RGB
            alpha = image_input[:, :, 3]
            rgb = image_input[:, :, :3]
            # Convert to grayscale
            gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
            # If drawn with white brush on black canvas:
            # Foreground where alpha > 0 and gray > 30
            img = np.where(alpha > 0, gray, 0.0)
        else:
            img = image_input.astype(np.float32)
    else:
        # PIL Image from file upload
        pil_img = image_input.convert("L")
        img = np.array(pil_img, dtype=np.float32)

    # Check if background is bright (white paper with dark ink)
    # Sample corners
    h, w = img.shape
    corner_mean = (img[:5, :5].mean() + img[:5, -5:].mean() + img[-5:, :5].mean() + img[-5:, -5:].mean()) / 4.0
    if corner_mean > 127.0:
        # Invert: black ink on white paper -> white ink on black background
        img = 255.0 - img

    # Threshold noise
    img[img < 25.0] = 0.0

    # If canvas is completely blank
    if img.max() == 0:
        blank = np.zeros((28, 28), dtype=np.float32)
        return blank, blank.reshape(1, 28, 28, 1)

    # Find bounding box of digit
    rows = np.any(img > 25, axis=1)
    cols = np.any(img > 25, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped = img[rmin : rmax + 1, cmin : cmax + 1]

    # Resize bounding box to fit within 20x20
    crop_h, crop_w = cropped.shape
    pil_crop = Image.fromarray(cropped.astype(np.uint8))

    if crop_h > crop_w:
        new_h = 20
        new_w = max(1, int(round((crop_w * 20.0) / crop_h)))
    else:
        new_w = 20
        new_h = max(1, int(round((crop_h * 20.0) / crop_w)))

    pil_resized = pil_crop.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_arr = np.array(pil_resized, dtype=np.float32)

    # Pad into 28x28 frame
    padded = np.zeros((28, 28), dtype=np.float32)
    pad_top = (28 - new_h) // 2
    pad_left = (28 - new_w) // 2
    padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized_arr

    # Center by mass
    centered = center_digit_by_mass(padded)

    # Normalize to [0.0, 1.0]
    norm_img = centered / 255.0
    norm_img = np.clip(norm_img, 0.0, 1.0)

    tensor_input = norm_img.reshape(1, 28, 28, 1)
    return norm_img, tensor_input


# Main App Layout
def main():
    st.markdown('<div class="main-title">Handwritten Digit Recognizer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Draw a digit or upload an image to predict its numerical class using a deep CNN trained on MNIST.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        model, model_path = load_trained_model()
        if model is not None:
            st.success(f"Model loaded: `{os.path.basename(model_path)}`")
        else:
            st.error("No trained model found! Please run `python train.py` first.")

        st.markdown("---")
        input_mode = st.radio("Input Method", ["✏️ Draw on Canvas", "📁 Upload Image"], index=0)

        if input_mode == "✏️ Draw on Canvas":
            stroke_width = st.slider("Brush Width", min_value=12, max_value=32, value=20, step=2)
            st.caption("Tip: Draw large and centered for best recognition.")
        else:
            stroke_width = 20

        st.markdown("---")
        st.markdown("### ℹ️ About the Model")
        st.markdown(
            """
            - **Dataset**: MNIST (60k train / 10k test)
            - **Architecture**: 4 Conv2D + BatchNormalization + Dropout + Dense
            - **Accuracy**: $\\ge 99.0\\%$
            - **Preprocessing**: Center-of-mass alignment & 20x20 aspect-ratio crop into 28x28.
            """
        )

    # Main columns: Left for input, Right for results
    col_input, col_output = st.columns([1.1, 1], gap="large")

    raw_image_data = None
    is_blank = True

    with col_input:
        if input_mode == "✏️ Draw on Canvas":
            st.subheader("Draw a Digit (0 - 9)")
            
            if "canvas_key" not in st.session_state:
                st.session_state["canvas_key"] = 0

            st_canvas = get_st_canvas()
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                return_image_data=True,
                key=f"canvas_{st.session_state['canvas_key']}",
            )

            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("🔄 Reset / Clear", use_container_width=True):
                    st.session_state["canvas_key"] += 1
                    st.rerun()

            if canvas_result is not None:
                try:
                    img_data = canvas_result.image_data
                    if img_data is not None and np.max(img_data[:, :, :3]) > 0:
                        raw_image_data = img_data
                        is_blank = False
                except Exception:
                    pass

        else:
            st.subheader("Upload an Image")
            uploaded_file = st.file_uploader("Upload digit image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                pil_image = Image.open(uploaded_file)
                st.image(pil_image, caption="Uploaded Image", width=200)
                raw_image_data = pil_image
                is_blank = False

    with col_output:
        st.subheader("Prediction & Analysis")

        if is_blank or raw_image_data is None:
            st.info("👈 Draw a digit on the canvas or upload an image to see the model's live prediction.")
        elif model is None:
            st.warning("Model file not found. Train the model using `python train.py` first.")
        else:
            # Preprocess
            proc_img, model_input = preprocess_image(raw_image_data)

            if proc_img.max() == 0:
                st.info("Please draw a clearer digit stroke.")
            else:
                # Predict
                preds = model.predict(model_input, verbose=0)[0]
                pred_digit = int(np.argmax(preds))
                confidence = float(preds[pred_digit]) * 100.0

                # Top Prediction Card
                st.markdown(
                    f"""
                    <div class="pred-card">
                        <div style="color: #94a3b8; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;">Predicted Digit</div>
                        <div class="pred-digit">{pred_digit}</div>
                        <div class="confidence-badge">{confidence:.2f}% Confidence</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # Confidence breakdown chart
                st.markdown("##### Class Probabilities (0–9)")
                chart_data = pd.DataFrame(
                    {
                        "Digit": [str(i) for i in range(10)],
                        "Probability": [float(p) for p in preds],
                    }
                )

                st.bar_chart(chart_data.set_index("Digit"), height=220)

                # Preprocessed Inspection
                with st.expander("🔍 View Preprocessed 28x28 Tensor"):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.image(
                            proc_img,
                            caption="28x28 Preprocessed Input",
                            width=112,
                            clamp=True,
                        )
                    with c2:
                        st.markdown(
                            f"""
                            - **Tensor Shape**: `(1, 28, 28, 1)`
                            - **Pixel Min / Max**: `{proc_img.min():.2f}` / `{proc_img.max():.2f}`
                            - **Top 3 Predictions**:
                            """
                        )
                        top3_idx = np.argsort(preds)[-3:][::-1]
                        top3_str = "".join(
                            [
                                f'<span class="top-pill"><b>{idx}</b>: {preds[idx]*100:.1f}%</span>'
                                for idx in top3_idx
                            ]
                        )
                        st.markdown(top3_str, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
