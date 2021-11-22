import os
from PIL import Image

dir_ = os.path.join(os.getcwd(), 'Dataset')
new_dir = os.path.join(os.getcwd(), 'resized_Dataset')
os.makedirs(new_dir, exist_ok=True)

files = os.listdir(dir_)

for file in files:
    class_dir = os.path.join(dir_, file)
    class_new_dir = os.path.join(new_dir, file)
    os.makedirs(class_new_dir, exist_ok=True)
    
    images = os.listdir(class_dir)
    
    for img in images:
        image = Image.open(os.path.join(class_dir, img))
        newsize = (128, 128)
        img_ = image.resize(newsize)
        img_path = os.path.join(class_new_dir, img)
        img_.save(img_path)
print("Created resized dataset!")
