# Handwritten Digit Classifier — ML Pipeline

**African Leadership University | BSE | Machine Learning Operations Summative**

A full end-to-end ML pipeline for classifying handwritten digits (0–9) using a custom CNN trained on real handwritten images.

---

## 📺 Video Demo
[https://youtu.be/3FdFttlRnMU](https://youtu.be/3FdFttlRnMU)

## 🌐 Live URL
https://mlop-kseh.onrender.com

---

## Project Description

This project builds a complete ML Operations pipeline around a **handwritten digit image classification** model. It covers:

- Custom CNN trained on real handwritten digit images (not MNIST)
- FastAPI REST API for predictions, uploads, and retraining
- Streamlit UI with model uptime, visualizations, upload, and retrain features
- Docker containerization and cloud deployment
- Locust load testing with latency/response time results

---

## Directory Structure

```
MLOP/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── locustfile.py
├── split_data.py
│
├── notebook/
│   └── handwritten_digit_classifier.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
│
├── api/
│   └── main.py
│
├── ui/
│   └── app.py
│
├── data/
│   ├── train/   (80% split — 10 class folders)
│   └── test/    (20% split — 10 class folders)
│
├── models/
│   └── handwritten_digit_model.h5
│
└── Handwritten_Dataset/   (original raw images)
    ├── zero/ … nine/
```

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd MLOP
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare data split
```bash
python split_data.py
```

### 4. Train the model
```bash
# Option A — via notebook
jupyter notebook notebook/handwritten_digit_classifier.ipynb

# Option B — via API (starts training in background)
uvicorn api.main:app --reload
# then POST http://localhost:8000/train
```

### 5. Run the API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Run the UI
```bash
streamlit run ui/app.py
```

---

## Docker Deployment

### Build and run (single container)
```bash
docker-compose up --build
```

### Scale API containers (for load testing comparison)
```bash
# 1 container
docker-compose up --build

# 2 containers
docker-compose up --build --scale api=2

# 3 containers
docker-compose up --build --scale api=3
```

- API: http://localhost:8000
- UI:  http://localhost:8501

---

## API Endpoints

| Method | Endpoint   | Description                          |
|--------|------------|--------------------------------------|
| GET    | /health    | Model uptime and status              |
| POST   | /predict   | Predict digit from uploaded image    |
| POST   | /upload    | Upload new images for a class        |
| POST   | /retrain   | Trigger model retraining             |
| POST   | /train     | Initial model training               |

---

## Load Testing with Locust

```bash
# Install locust
pip install locust

# Run load test (API must be running)
locust -f locustfile.py --host http://localhost:8000
```

Open http://localhost:8089 in your browser to configure users and spawn rate.

### Results Summary

| Containers | Users | Avg Latency (ms) | 95th Percentile (ms) | RPS  |
|------------|-------|------------------|----------------------|------|
| 1          | 10    | 473              | 1100                 | 5.6  |
| 2          | 10    | 856              | 2600                 | 6.3  |
| 3          | 10    | 900              | 3000                 | 6.4  |

#### 1 Container — Detailed Breakdown

| Type | Endpoint   | # Requests | Avg (ms) | 95%ile (ms) | 99%ile (ms) | Min (ms) | Max (ms) |
|------|------------|------------|----------|-------------|-------------|----------|----------|
| GET  | /health    | 164        | 403      | 1100        | 3800        | 3        | 3858     |
| POST | /predict   | 811        | 487      | 1100        | 2300        | 116      | 3809     |
| —    | Aggregated | 975        | 473      | 1100        | 2500        | 3        | 3858     |

- Failures: 0%
- Total Requests: 975
- RPS: 5.6

#### 2 Containers — Detailed Breakdown

| Type | Endpoint   | # Requests | Avg (ms) | 95%ile (ms) | 99%ile (ms) | Min (ms) | Max (ms) |
|------|------------|------------|----------|-------------|-------------|----------|----------|
| GET  | /health    | 133        | 634      | 2100        | 3400        | 3        | 3577     |
| POST | /predict   | 759        | 895      | 2600        | 4200        | 121      | 4992     |
| —    | Aggregated | 892        | 856      | 2600        | 4100        | 3        | 4992     |

- Failures: 0%
- Total Requests: 892
- RPS: 6.3

#### 3 Containers — Detailed Breakdown

| Type | Endpoint   | # Requests | Avg (ms) | 95%ile (ms) | 99%ile (ms) | Min (ms) | Max (ms) |
|------|------------|------------|----------|-------------|-------------|----------|----------|
| GET  | /health    | 143        | 762      | 3000        | 5200        | 4        | 6926     |
| POST | /predict   | 730        | 927      | 3000        | 4900        | 127      | 6729     |
| —    | Aggregated | 873        | 900      | 3000        | 4900        | 4        | 6926     |

- Failures: 0%
- Total Requests: 873
- RPS: 6.4

---

## Model Details

- Architecture: MobileNetV2 (transfer learning, top 30 layers fine-tuned)
- Input: 128×128 RGB images (grayscale → inverted → RGB → scaled to [-1, 1])
- Output: 10 classes (zero through nine)
- Training: 80/20 split, 10× augmentation, two-phase training (frozen base then fine-tune)
- Saved as: `models/handwritten_digit_model.h5`

---

## Dataset

- 10 classes of handwritten digits (0–9)
- ~20 real handwritten images per class (scanned from paper)
- Duplicate/copy files removed during preprocessing
- Split: ~16 train / ~4 test per class
