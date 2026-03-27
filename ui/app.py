import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time

API_URL = os.getenv('API_URL', 'http://localhost:8000')

st.set_page_config(page_title='Handwritten Digit Classifier', layout='wide')
st.title('✍️ Handwritten Digit Classifier')

# ── sidebar navigation ────────────────────────────────────────────────────────
page = st.sidebar.radio('Navigate', ['Model Status', 'Predict', 'Visualizations', 'Upload & Retrain'])

# ── 1. Model Status ───────────────────────────────────────────────────────────
if page == 'Model Status':
    st.header('Model Up-time & Health')
    try:
        r = requests.get(f'{API_URL}/health', timeout=5)
        data = r.json()
        col1, col2, col3 = st.columns(3)
        col1.metric('Status', data.get('status', 'unknown').upper())
        col2.metric('Uptime', data.get('uptime', 'N/A'))
        col3.metric('Model Ready', '✅' if data.get('model_ready') else '❌')
        st.info(f"Retrain Status: **{data.get('retrain_status', 'N/A')}**")
        if data.get('last_trained'):
            st.success(f"Last trained: {data['last_trained']}")
    except Exception as e:
        st.error(f'Cannot reach API: {e}')

    st.divider()
    st.subheader('Initial Training')
    st.write('If the model has not been trained yet, click below to train it.')
    if st.button('🚀 Train Model'):
        r = requests.post(f'{API_URL}/train', timeout=10)
        st.success(r.json().get('message'))

# ── 2. Predict ────────────────────────────────────────────────────────────────
elif page == 'Predict':
    st.header('Predict a Single Image')
    uploaded = st.file_uploader('Upload a handwritten digit image', type=['jpg', 'jpeg', 'png'])
    if uploaded:
        st.image(uploaded, caption='Uploaded Image', width=200)
        if st.button('🔍 Predict'):
            with st.spinner('Predicting...'):
                r = requests.post(
                    f'{API_URL}/predict',
                    files={'file': (uploaded.name, uploaded.getvalue(), 'image/jpeg')},
                    timeout=15
                )
            if r.status_code == 200:
                result = r.json()
                st.success(f"Predicted: **{result['predicted_class'].upper()}**  |  Confidence: **{result['confidence']*100:.1f}%**")
                st.caption(f"Latency: {result.get('latency_ms', 'N/A')} ms")

                probs = result['all_probabilities']
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.barh(list(probs.keys()), [v * 100 for v in probs.values()], color='steelblue')
                ax.set_xlabel('Confidence (%)')
                ax.set_title('Class Probabilities')
                st.pyplot(fig)
            else:
                st.error(f'Error: {r.text}')

# ── 3. Visualizations ─────────────────────────────────────────────────────────
elif page == 'Visualizations':
    st.header('Dataset Visualizations')

    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'train')
    CLASSES = ['eight', 'five', 'four', 'nine', 'one', 'seven', 'six', 'three', 'two', 'zero']

    # Feature 1: Class distribution
    st.subheader('1. Class Distribution')
    counts = {}
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        counts[cls] = len(os.listdir(cls_dir)) if os.path.isdir(cls_dir) else 0
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.bar(counts.keys(), counts.values(), color='coral')
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Training Images per Class')
    plt.xticks(rotation=45)
    st.pyplot(fig1)
    st.caption('Story: The dataset is roughly balanced (~16 images/class), meaning the model is not biased toward any digit.')

    # Feature 2: Sample images per class
    st.subheader('2. Sample Images per Class')
    fig2, axes = plt.subplots(2, 5, figsize=(12, 5))
    for ax, cls in zip(axes.flatten(), CLASSES):
        cls_dir = os.path.join(DATA_DIR, cls)
        if os.path.isdir(cls_dir):
            imgs = [f for f in os.listdir(cls_dir) if f.endswith('.jpg')]
            if imgs:
                img = mpimg.imread(os.path.join(cls_dir, imgs[0]))
                ax.imshow(img)
        ax.set_title(cls)
        ax.axis('off')
    st.pyplot(fig2)
    st.caption('Story: Each class contains real handwritten digits scanned from paper, showing natural variation in stroke width and style.')

    # Feature 3: Average pixel intensity per class
    st.subheader('3. Average Pixel Intensity per Class')
    import numpy as np
    from PIL import Image as PILImage
    avg_brightness = {}
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        vals = []
        for fname in os.listdir(cls_dir):
            if fname.endswith('.jpg'):
                img = np.array(PILImage.open(os.path.join(cls_dir, fname)).convert('L').resize((64, 64)), dtype=np.float32)
                vals.append(img.mean())
        avg_brightness[cls] = np.mean(vals) if vals else 0

    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.bar(avg_brightness.keys(), avg_brightness.values(), color='mediumseagreen')
    ax3.set_ylabel('Avg Pixel Intensity (0–255)')
    ax3.set_title('Average Brightness per Class')
    plt.xticks(rotation=45)
    st.pyplot(fig3)
    st.caption('Story: Digits like "1" tend to be brighter (more white space) while "8" and "0" are darker due to more ink coverage — this is a useful signal for the model.')

# ── 4. Upload & Retrain ───────────────────────────────────────────────────────
elif page == 'Upload & Retrain':
    st.header('Upload New Data & Retrain')

    CLASSES = ['eight', 'five', 'four', 'nine', 'one', 'seven', 'six', 'three', 'two', 'zero']
    cls = st.selectbox('Select digit class for uploaded images', CLASSES)
    files = st.file_uploader('Upload images (JPG)', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if files and st.button('📤 Upload Images'):
        file_tuples = [('files', (f.name, f.getvalue(), 'image/jpeg')) for f in files]
        r = requests.post(f'{API_URL}/upload?cls={cls}', files=file_tuples, timeout=30)
        if r.status_code == 200:
            st.success(f"Uploaded {len(r.json()['uploaded'])} images for class '{cls}'")
        else:
            st.error(r.text)

    st.divider()
    st.subheader('Trigger Retraining')
    st.write('Click below to retrain the model using all current training data (including newly uploaded images).')
    if st.button('🔄 Retrain Model'):
        r = requests.post(f'{API_URL}/retrain', timeout=10)
        st.info(r.json().get('message'))
        st.write('Check **Model Status** page to monitor progress.')
