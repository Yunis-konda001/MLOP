import os
import sys
import time
import shutil
import tempfile
import threading
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction import predict
from src.model import train, load_model, MODEL_PATH

app = FastAPI(title='Handwritten Digit Classifier API')

@app.middleware('http')
async def log_requests(request, call_next):
    print(f'[REQUEST] {request.method} {request.url}')
    try:
        response = await call_next(request)
        print(f'[RESPONSE] status={response.status_code}')
        return response
    except Exception as e:
        import traceback
        print(f'[MIDDLEWARE ERROR] {e}')
        traceback.print_exc()
        raise

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── shared state ──────────────────────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()
_start_time = datetime.utcnow()
_retrain_status = {'status': 'idle', 'last_trained': None}

UPLOAD_DIR = os.path.join('data', 'retrain')

# Load model at startup
if os.path.exists(MODEL_PATH):
    print(f'[INFO] Loading model from {MODEL_PATH}...')
    _model = load_model(MODEL_PATH)
    print('[INFO] Model loaded OK')
else:
    print(f'[WARNING] Model not found at {MODEL_PATH}')


def get_model():
    if _model is None:
        raise HTTPException(status_code=503, detail='Model not trained yet. POST /train first.')
    return _model


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get('/health')
def health():
    uptime = str(datetime.utcnow() - _start_time).split('.')[0]
    return {
        'status': 'ok',
        'uptime': uptime,
        'model_ready': os.path.exists(MODEL_PATH),
        'retrain_status': _retrain_status['status'],
        'last_trained': _retrain_status['last_trained'],
    }


@app.post('/predict')
async def predict_endpoint(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1] or '.jpg'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        t0 = time.time()
        result = predict(tmp_path, model=get_model())
        result['latency_ms'] = round((time.time() - t0) * 1000, 2)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post('/upload')
async def upload_data(cls: str, files: list[UploadFile] = File(...)):
    """Upload new images for a given class to trigger retraining later."""
    cls_dir = os.path.join(UPLOAD_DIR, cls)
    os.makedirs(cls_dir, exist_ok=True)
    saved = []
    for file in files:
        dest = os.path.join(cls_dir, file.filename)
        with open(dest, 'wb') as f:
            f.write(await file.read())
        saved.append(file.filename)
    return {'uploaded': saved, 'class': cls}


def _retrain_job():
    global _model, _retrain_status
    _retrain_status['status'] = 'training'
    try:
        # merge uploaded data into train dir
        if os.path.exists(UPLOAD_DIR):
            for cls in os.listdir(UPLOAD_DIR):
                src = os.path.join(UPLOAD_DIR, cls)
                dst = os.path.join('data', 'train', cls)
                os.makedirs(dst, exist_ok=True)
                for f in os.listdir(src):
                    shutil.copy(os.path.join(src, f), os.path.join(dst, f))

        new_model = train()
        with _model_lock:
            _model = new_model
        _retrain_status['status'] = 'idle'
        _retrain_status['last_trained'] = datetime.utcnow().isoformat()
    except Exception as e:
        _retrain_status['status'] = f'error: {e}'


@app.post('/retrain')
def retrain(background_tasks: BackgroundTasks):
    if _retrain_status['status'] == 'training':
        return {'message': 'Retraining already in progress.'}
    background_tasks.add_task(_retrain_job)
    return {'message': 'Retraining started in background.'}


@app.post('/train')
def initial_train(background_tasks: BackgroundTasks):
    background_tasks.add_task(_retrain_job)
    return {'message': 'Training started in background.'}
