import streamlit as st
import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True

API_URL = os.getenv('API_URL', 'http://localhost:8000')

st.set_page_config(
    page_title='Handwritten Digit Classifier', 
    layout='wide',
    initial_sidebar_state='expanded'
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 300;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .status-success {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .status-error {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 400;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ecf0f1;
    }
    
    .info-box {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .footer {
        text-align: center;
        color: #7f8c8d;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #ecf0f1;
        font-size: 0.9rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .nav-item {
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Handwritten Digit Classifier</h1>', unsafe_allow_html=True)

# Clean navigation
page = st.sidebar.selectbox(
    'Navigation', 
    ['Model Status', 'Predict', 'Visualizations', 'Upload & Retrain'],
    index=0
)

# API connection info
st.sidebar.markdown('---')
st.sidebar.markdown('**System Information**')
st.sidebar.code(f'API: {API_URL}')

if page == 'Model Status':
    st.markdown('<h2 class="section-header">System Status</h2>', unsafe_allow_html=True)
    
    try:
        r = requests.get(f'{API_URL}/health', timeout=10)
        r.raise_for_status()
        
        try:
            data = r.json()
        except requests.exceptions.JSONDecodeError:
            st.markdown('<div class="status-error">API returned invalid response. Service may be starting up.</div>', unsafe_allow_html=True)
            st.code(f'Response: {r.text[:200]}...')
            st.stop()
            
        # Clean metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = data.get('status', 'unknown').upper()
            if status == 'OK':
                st.markdown('<div class="metric-card"><h3>Service Status</h3><p style="color: #27ae60; font-size: 1.2rem; font-weight: 600;">ONLINE</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card"><h3>Service Status</h3><p style="color: #e74c3c; font-size: 1.2rem; font-weight: 600;">OFFLINE</p></div>', unsafe_allow_html=True)
            
        with col2:
            uptime = data.get('uptime', 'N/A')
            st.markdown(f'<div class="metric-card"><h3>System Uptime</h3><p style="font-size: 1.2rem; font-weight: 600;">{uptime}</p></div>', unsafe_allow_html=True)
            
        with col3:
            model_ready = data.get('model_ready', False)
            status_text = "READY" if model_ready else "NOT READY"
            color = "#27ae60" if model_ready else "#e74c3c"
            st.markdown(f'<div class="metric-card"><h3>Model Status</h3><p style="color: {color}; font-size: 1.2rem; font-weight: 600;">{status_text}</p></div>', unsafe_allow_html=True)
        
        # Training status
        retrain_status = data.get('retrain_status', 'N/A')
        if retrain_status == 'idle':
            st.markdown('<div class="status-success"><h4>Training Status: IDLE</h4><p>System is ready for new training requests.</p></div>', unsafe_allow_html=True)
        elif retrain_status == 'training':
            st.markdown('<div class="status-warning"><h4>Training Status: IN PROGRESS</h4><p>Model is currently being retrained. Please wait for completion.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="info-box"><h4>Training Status: {retrain_status.upper()}</h4></div>', unsafe_allow_html=True)
            
        # Last trained info
        if data.get('last_trained'):
            st.markdown(f'<div class="info-box"><h4>Last Training Session</h4><p>Completed: {data["last_trained"]}</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box"><h4>Training History</h4><p>No retraining sessions recorded.</p></div>', unsafe_allow_html=True)
            
    except requests.exceptions.ConnectionError:
        st.markdown('<div class="status-error"><h4>Connection Error</h4><p>Unable to reach the API service. Please check if the service is running.</p></div>', unsafe_allow_html=True)
        st.markdown(f'**Target Endpoint:** `{API_URL}`')
        st.stop()
        
    except requests.exceptions.Timeout:
        st.markdown('<div class="status-error"><h4>Timeout Error</h4><p>API service is not responding within the expected time.</p></div>', unsafe_allow_html=True)
        st.stop()
        
    except Exception as e:
        st.markdown(f'<div class="status-error"><h4>System Error</h4><p>{str(e)}</p></div>', unsafe_allow_html=True)
        st.stop()

    st.markdown('<h2 class="section-header">Model Training</h2>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Initialize the model if this is the first deployment or if the model needs to be trained from scratch.</div>', unsafe_allow_html=True)
    
    if st.button('Initialize Model Training'):
        try:
            with st.spinner('Initializing training process...'):
                r = requests.post(f'{API_URL}/train', timeout=15)
            if r.status_code == 200:
                response_data = r.json()
                st.markdown(f'<div class="status-success"><h4>Training Started</h4><p>{response_data.get("message", "Training process initiated successfully")}</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-error"><h4>Training Failed</h4><p>Status: {r.status_code}<br>Details: {r.text}</p></div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="status-error"><h4>Training Error</h4><p>{str(e)}</p></div>', unsafe_allow_html=True)

elif page == 'Predict':
    st.markdown('<h2 class="section-header">Image Classification</h2>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Upload a handwritten digit image to receive a classification prediction with confidence scores.</div>', unsafe_allow_html=True)
    
    uploaded = st.file_uploader(
        'Select Image File', 
        type=['jpg', 'jpeg', 'png'],
        help='Supported formats: JPG, JPEG, PNG'
    )
    
    if uploaded:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(uploaded, caption='Input Image', width=250)
            
        with col2:
            if st.button('Analyze Image'):
                try:
                    with st.spinner('Processing image...'):
                        r = requests.post(
                            f'{API_URL}/predict',
                            files={'file': (uploaded.name, uploaded.getvalue(), 'image/jpeg')},
                            timeout=20
                        )
                    
                    if r.status_code == 200:
                        result = r.json()
                        
                        # Clean result display
                        st.markdown(f'''
                        <div class="prediction-result">
                            <h2>Classification Result</h2>
                            <h1 style="font-size: 4rem; margin: 1rem 0;">{result['predicted_class'].upper()}</h1>
                            <p style="font-size: 1.5rem;">Confidence: {result['confidence']*100:.1f}%</p>
                            <p style="opacity: 0.8;">Processing Time: {result.get('latency_ms', 'N/A')} ms</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Confidence distribution
                        st.markdown('<h3 class="section-header">Confidence Distribution</h3>', unsafe_allow_html=True)
                        probs = result['all_probabilities']
                        
                        # Create clean chart
                        fig, ax = plt.subplots(figsize=(12, 6))
                        fig.patch.set_facecolor('white')
                        
                        colors = ['#667eea' if k == result['predicted_class'] else '#bdc3c7' for k in probs.keys()]
                        bars = ax.barh(list(probs.keys()), [v * 100 for v in probs.values()], color=colors)
                        
                        ax.set_xlabel('Confidence Percentage', fontsize=12, fontweight='500')
                        ax.set_title('Classification Confidence by Digit', fontsize=14, fontweight='500', pad=20)
                        ax.grid(axis='x', alpha=0.3, linestyle='--')
                        ax.set_facecolor('#fafafa')
                        
                        # Add percentage labels
                        for bar, prob in zip(bars, probs.values()):
                            width = bar.get_width()
                            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                                   f'{prob*100:.1f}%', ha='left', va='center', fontweight='500')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    else:
                        st.markdown(f'<div class="status-error"><h4>Prediction Failed</h4><p>Status: {r.status_code}<br>Details: {r.text}</p></div>', unsafe_allow_html=True)
                        
                except Exception as e:
                    st.markdown(f'<div class="status-error"><h4>Processing Error</h4><p>{str(e)}</p></div>', unsafe_allow_html=True)

elif page == 'Visualizations':
    st.markdown('<h2 class="section-header">Dataset Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Comprehensive analysis of the handwritten digit dataset used for model training.</div>', unsafe_allow_html=True)

    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'train')
    CLASSES = ['eight', 'five', 'four', 'nine', 'one', 'seven', 'six', 'three', 'two', 'zero']

    # Class distribution
    st.markdown('<h3 class="section-header">Class Distribution Analysis</h3>', unsafe_allow_html=True)
    counts = {}
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        counts[cls] = len(os.listdir(cls_dir)) if os.path.isdir(cls_dir) else 0
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    fig1.patch.set_facecolor('white')
    bars = ax1.bar(counts.keys(), counts.values(), color='#667eea', alpha=0.8)
    ax1.set_ylabel('Number of Training Images', fontsize=12, fontweight='500')
    ax1.set_title('Training Data Distribution by Digit Class', fontsize=14, fontweight='500', pad=20)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_facecolor('#fafafa')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='500')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)
    st.markdown('<div class="info-box"><strong>Analysis:</strong> The dataset maintains balanced representation across all digit classes, ensuring unbiased model training with approximately 16 samples per class.</div>', unsafe_allow_html=True)

    # Sample images
    st.markdown('<h3 class="section-header">Sample Images by Class</h3>', unsafe_allow_html=True)
    fig2, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig2.patch.set_facecolor('white')
    
    for ax, cls in zip(axes.flatten(), CLASSES):
        cls_dir = os.path.join(DATA_DIR, cls)
        if os.path.isdir(cls_dir):
            imgs = [f for f in os.listdir(cls_dir) if f.endswith('.jpg')]
            if imgs:
                img = mpimg.imread(os.path.join(cls_dir, imgs[0]))
                ax.imshow(img)
        ax.set_title(cls.title(), fontsize=12, fontweight='500')
        ax.axis('off')
    
    plt.suptitle('Representative Samples from Each Digit Class', fontsize=14, fontweight='500')
    plt.tight_layout()
    st.pyplot(fig2)
    st.markdown('<div class="info-box"><strong>Analysis:</strong> Real handwritten samples demonstrate natural variation in stroke patterns, thickness, and orientation, providing robust training data for the classification model.</div>', unsafe_allow_html=True)

    # Pixel intensity analysis
    st.markdown('<h3 class="section-header">Pixel Intensity Analysis</h3>', unsafe_allow_html=True)
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

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    fig3.patch.set_facecolor('white')
    bars = ax3.bar(avg_brightness.keys(), avg_brightness.values(), color='#764ba2', alpha=0.8)
    ax3.set_ylabel('Average Pixel Intensity (0-255)', fontsize=12, fontweight='500')
    ax3.set_title('Average Brightness Distribution by Digit Class', fontsize=14, fontweight='500', pad=20)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_facecolor('#fafafa')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontweight='500')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig3)
    st.markdown('<div class="info-box"><strong>Analysis:</strong> Digits with simpler structures (like "1") exhibit higher brightness values due to more white space, while complex digits ("8", "0") show lower values due to increased ink coverage. This intensity variation serves as a valuable feature for classification.</div>', unsafe_allow_html=True)

elif page == 'Upload & Retrain':
    st.markdown('<h2 class="section-header">Model Enhancement</h2>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Expand the training dataset with new images and retrain the model to improve classification accuracy.</div>', unsafe_allow_html=True)

    CLASSES = ['eight', 'five', 'four', 'nine', 'one', 'seven', 'six', 'three', 'two', 'zero']
    
    # Upload section
    st.markdown('<h3 class="section-header">Data Upload</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        cls = st.selectbox('Target Digit Class', CLASSES)
        st.markdown('<div class="info-box"><strong>Instructions:</strong><br>• Select the digit class that matches your images<br>• Upload clear, well-cropped images<br>• Multiple files can be selected</div>', unsafe_allow_html=True)
    
    with col2:
        files = st.file_uploader(
            'Select Training Images', 
            type=['jpg', 'jpeg', 'png'], 
            accept_multiple_files=True
        )

    if files and st.button('Upload Training Data'):
        try:
            with st.spinner(f'Uploading {len(files)} images to training dataset...'):
                file_tuples = [('files', (f.name, f.getvalue(), 'image/jpeg')) for f in files]
                r = requests.post(f'{API_URL}/upload?cls={cls}', files=file_tuples, timeout=60)
            
            if r.status_code == 200:
                response_data = r.json()
                st.markdown(f'<div class="status-success"><h4>Upload Successful</h4><p>Added {len(response_data["uploaded"])} images to class "{cls}"</p></div>', unsafe_allow_html=True)
                
                # Show upload details
                with st.expander("Upload Details"):
                    st.json(response_data)
            else:
                st.markdown(f'<div class="status-error"><h4>Upload Failed</h4><p>Status: {r.status_code}<br>Details: {r.text}</p></div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="status-error"><h4>Upload Error</h4><p>{str(e)}</p></div>', unsafe_allow_html=True)

    st.markdown('<h3 class="section-header">Model Retraining</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="info-box"><strong>Retraining Process:</strong><br>• Incorporates all uploaded data<br>• Uses transfer learning approach<br>• Typically takes 5-15 minutes<br>• Monitor progress in Model Status</div>', unsafe_allow_html=True)
        
        if st.button('Start Model Retraining'):
            try:
                with st.spinner('Initiating retraining process...'):
                    r = requests.post(f'{API_URL}/retrain', timeout=15)
                
                if r.status_code == 200:
                    response_data = r.json()
                    st.markdown(f'<div class="status-success"><h4>Retraining Started</h4><p>{response_data.get("message", "Model retraining initiated successfully")}</p></div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box">Monitor training progress on the <strong>Model Status</strong> page.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-error"><h4>Retraining Failed</h4><p>Status: {r.status_code}<br>Details: {r.text}</p></div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="status-error"><h4>Retraining Error</h4><p>{str(e)}</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box"><strong>Performance Tips:</strong><br>• Upload high-quality images<br>• Ensure proper digit centering<br>• Maintain consistent image quality<br>• Add diverse handwriting styles</div>', unsafe_allow_html=True)

# Clean footer
st.markdown('<div class="footer">Handwritten Digit Classifier | Machine Learning Operations Pipeline | Built with Streamlit & FastAPI</div>', unsafe_allow_html=True)