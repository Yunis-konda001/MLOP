import os, shutil, random
random.seed(42)
src = 'Handwritten_Dataset'
for cls in os.listdir(src):
    cls_path = os.path.join(src, cls)
    if not os.path.isdir(cls_path):
        continue
    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith('.jpg') and '- Copy' not in f]
    random.shuffle(imgs)
    split = int(len(imgs) * 0.8)
    train_imgs, test_imgs = imgs[:split], imgs[split:]
    os.makedirs(os.path.join('data', 'train', cls), exist_ok=True)
    os.makedirs(os.path.join('data', 'test', cls), exist_ok=True)
    for f in train_imgs:
        shutil.copy(os.path.join(cls_path, f), os.path.join('data', 'train', cls, f))
    for f in test_imgs:
        shutil.copy(os.path.join(cls_path, f), os.path.join('data', 'test', cls, f))
    print(f'{cls}: {len(train_imgs)} train, {len(test_imgs)} test')
print('Done.')
