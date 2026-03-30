import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time

API_URL = os.getenv('API_URL', 'http://localhost:8000')

st.set_page_config(
    page_title='Handwritten Digit Classifier', 
    layout='wide',
    page_icon='🔢',
    initial_sidebar_state='expanded'
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Handwritten Digit Classifier</h1>', unsafe_allow_html=True)

# Navigation with better styling
page = st.sidebar.radio(
    '📋 Navigation', 
    ['🏠 Model Status', '🔍 Predict', '📊 Visualizations', '📤 Upload & Retrain'],
    index=0
)

# Add API connection info in sidebar
st.sidebar.markdown('---')
st.sidebar.markdown(f'**API Endpoint:** `{API_URL}`')

if page == '🏠 Model Status':
    st.header('🏠 Model Status & Health')
    
    # Test API connection first
    try:
        r = requests.get(f'{API_URL}/health', timeout=10)
        r.raise_for_status()  # Raises an HTTPError for bad responses
        
        try:
            data = r.json()
        except requests.exceptions.JSONDecodeError:
            st.error('⚠️ API returned invalid JSON response. Check if the API service is running correctly.')
            st.code(f'Response: {r.text[:200]}...')
            st.stop()
            
        # Display metrics in a nice layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = data.get('status', 'unknown').upper()
            status_color = '🟢' if status == 'OK' else '🔴'
            st.metric('Service Status', f'{status_color} {status}')
            
        with col2:
            uptime = data.get('uptime', 'N/A')
            st.metric('Uptime', f'⏱️ {uptime}')
            
        with col3:
            model_ready = data.get('model_ready', False)
            ready_icon = '✅' if model_ready else '❌'
            st.metric('Model Ready', f'{ready_icon} {"Yes" if model_ready else "No"}')
        
        # Retrain status
        retrain_status = data.get('retrain_status', 'N/A')
        if retrain_status == 'idle':
            st.success(f'🔄 Retrain Status: **{retrain_status.title()}**')
        elif retrain_status == 'training':
            st.warning(f'🔄 Retrain Status: **{retrain_status.title()}** (Please wait...)')
        else:
            st.info(f'🔄 Retrain Status: **{retrain_status}**')
            
        # Last trained info
        if data.get('last_trained'):
            st.success(f'📅 Last trained: **{data["last_trained"]}**')
        else:
            st.info('📅 Model has not been retrained yet')
            
    except requests.exceptions.ConnectionError:
        st.error('🚫 **Connection Error**: Cannot reach the API service.')
        st.markdown(f'**Trying to connect to:** `{API_URL}`')
        st.markdown('**Possible causes:**')
        st.markdown('- API service is not running')
        st.markdown('- Wrong API URL in environment variables')
        st.markdown('- Network connectivity issues')
        st.stop()
        
    except requests.exceptions.Timeout:
        st.error('⏰ **Timeout Error**: API service is not responding.')
        st.stop()
        
    except requests.exceptions.HTTPError as e:
        st.error(f'🚫 **HTTP Error**: {e}')
        st.stop()
        
    except Exception as e:
        st.error(f'❌ **Unexpected Error**: {e}')
        st.stop()

    st.divider()
    st.subheader('🚀 Initial Training')
    st.write('If the model has not been trained yet, click below to train it.')
    
    if st.button('🚀 Train Model', type='primary'):
        try:
            with st.spinner('Starting training...'):
                r = requests.post(f'{API_URL}/train', timeout=15)
            if r.status_code == 200:
                response_data = r.json()
                st.success(f'✅ {response_data.get("message", "Training started successfully")}')
            else:
                st.error(f'❌ Training failed: {r.status_code} - {r.text}')
        except Exception as e:
            st.error(f'❌ Failed to start training: {e}')

elif page == '🔍 Predict':
    st.header('🔍 Single Image Prediction')
    st.write('Upload a handwritten digit image to get a prediction from the trained model.')
    
    uploaded = st.file_uploader(
        'Choose an image file', 
        type=['jpg', 'jpeg', 'png'],
        help='Upload a clear image of a handwritten digit (0-9)'
    )
    
    if uploaded:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(uploaded, caption='Uploaded Image', width=200)
            
        with col2:
            if st.button('🔍 Predict Digit', type='primary'):
                try:
                    with st.spinner('Analyzing image...'):
                        r = requests.post(
                            f'{API_URL}/predict',
                            files={'file': (uploaded.name, uploaded.getvalue(), 'image/jpeg')},
                            timeout=20
                        )
                    
                    if r.status_code == 200:
                        result = r.json()
                        
                        # Display result prominently
                        st.success(f"🎯 **Predicted Digit: {result['predicted_class'].upper()}**")
                        st.info(f"🎯 **Confidence: {result['confidence']*100:.1f}%**")
                        st.caption(f"⚡ Processing time: {result.get('latency_ms', 'N/A')} ms")
                        
                        # Show probability chart
                        st.subheader('📊 Confidence Distribution')
                        probs = result['all_probabilities']
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        colors = ['#1f77b4' if k == result['predicted_class'] else '#d3d3d3' for k in probs.keys()]
                        bars = ax.barh(list(probs.keys()), [v * 100 for v in probs.values()], color=colors)
                        ax.set_xlabel('Confidence (%)')
                        ax.set_title('Prediction Confidence for Each Digit')
                        ax.grid(axis='x', alpha=0.3)
                        
                        # Add percentage labels on bars
                        for bar, prob in zip(bars, probs.values()):
                            width = bar.get_width()
                            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                                   f'{prob*100:.1f}%', ha='left', va='center')
                        
                        st.pyplot(fig)
                        
                    else:
                        st.error(f'❌ Prediction failed: {r.status_code} - {r.text}')
                        
                except Exception as e:
                    st.error(f'❌ Prediction error: {e}')

elif page == '📊 Visualizations':
    st.header('📊 Dataset Analysis & Insights')
    st.write('Explore the characteristics of our handwritten digit dataset.')

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

elif page == '📤 Upload & Retrain':
    st.header('📤 Upload New Data & Retrain Model')
    st.write('Add new training images and retrain the model to improve its performance.')

    CLASSES = ['eight', 'five', 'four', 'nine', 'one', 'seven', 'six', 'three', 'two', 'zero']
    
    # Upload section
    st.subheader('📁 Upload New Training Images')
    col1, col2 = st.columns([1, 2])
    
    with col1:
        cls = st.selectbox('Select digit class', CLASSES, help='Choose which digit these images represent')
    
    with col2:
        files = st.file_uploader(
            'Choose image files', 
            type=['jpg', 'jpeg', 'png'], 
            accept_multiple_files=True,
            help='Upload multiple images of the same digit'
        )

    if files and st.button('📤 Upload Images', type='primary'):
        try:
            with st.spinner(f'Uploading {len(files)} images...'):
                file_tuples = [('files', (f.name, f.getvalue(), 'image/jpeg')) for f in files]
                r = requests.post(f'{API_URL}/upload?cls={cls}', files=file_tuples, timeout=60)
            
            if r.status_code == 200:
                response_data = r.json()
                st.success(f"✅ Successfully uploaded {len(response_data['uploaded'])} images for class '{cls}'")
                st.json(response_data)  # Show upload details
            else:
                st.error(f'❌ Upload failed: {r.status_code} - {r.text}')
        except Exception as e:
            st.error(f'❌ Upload error: {e}')

    st.divider()
    
    # Retrain section
    st.subheader('🔄 Trigger Model Retraining')
    st.write('Retrain the model using all current training data (including newly uploaded images).')
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button('🔄 Start Retraining', type='primary'):
            try:
                with st.spinner('Starting retraining process...'):
                    r = requests.post(f'{API_URL}/retrain', timeout=15)
                
                if r.status_code == 200:
                    response_data = r.json()
                    st.success(f'✅ {response_data.get("message", "Retraining started successfully")}')
                    st.info('📋 Check the **Model Status** page to monitor training progress.')
                else:
                    st.error(f'❌ Retrain failed: {r.status_code} - {r.text}')
            except Exception as e:
                st.error(f'❌ Retrain error: {e}')
    
    with col2:
        st.info('💡 **Tip**: Retraining can take 5-15 minutes depending on the amount of data and server resources.')

# Footer
st.markdown('---')
st.markdown(
    '<div style="text-align: center; color: #666; padding: 1rem;">' +
    'Handwritten Digit Classifier | Built with Streamlit & FastAPI | MLOps Pipeline' +
    '</div>', 
    unsafe_allow_html=True
)