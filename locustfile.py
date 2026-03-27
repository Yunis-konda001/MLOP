import os
import random
from locust import HttpUser, task, between

# Collect all test images
TEST_DIR = os.path.join(os.path.dirname(__file__), 'data', 'test')
_images = []
if os.path.exists(TEST_DIR):
    for cls in os.listdir(TEST_DIR):
        cls_dir = os.path.join(TEST_DIR, cls)
        for fname in os.listdir(cls_dir):
            if fname.endswith('.jpg'):
                _images.append(os.path.join(cls_dir, fname))


class DigitClassifierUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(5)
    def predict(self):
        if not _images:
            return
        img_path = random.choice(_images)
        with open(img_path, 'rb') as f:
            self.client.post(
                '/predict',
                files={'file': (os.path.basename(img_path), f, 'image/jpeg')},
                name='/predict'
            )

    @task(1)
    def health_check(self):
        self.client.get('/health', name='/health')
