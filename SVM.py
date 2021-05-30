import os
from PIL import Image
import numpy as np
# import cv2
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score


# 加载训练集 / 验证集数据 + 标签
def load_data(image_base, label_file):
    images = []
    labels = []
    for img_path in tqdm(os.listdir(image_base)):
        try:
            img = np.asarray(Image.open(os.path.join(image_base, img_path)).resize([100, 100], Image.ANTIALIAS),
                             dtype=np.int32).flatten()
            images.append((int(os.path.splitext(img_path)[0]), img))
        finally:
            pass

    images = [img for num, img in sorted(images, key=lambda x: x[0])]

    with open(label_file, 'r') as f:
        lines = f.read().split('\n')[:-1]
        for line in lines:
            num, id = line.split()
            labels.append((num, id))

    labels = [id for num, id in sorted(labels, key=lambda x: x[0])]

    return np.asarray(images), np.asarray(labels)


def get_model():
    images_base = 'data/extract_images2'
    label_file = 'data/extract_number_to_tag3.txt'
    train_data, train_labels = load_data(images_base, label_file)
    print(train_data.shape)
    print(train_labels.shape)

    model = LinearSVC()
    model.fit(train_data, train_labels)

    print('finish fit')

    return model


if __name__ == '__main__':
    model = get_model()

    images_base = 'data/extract_images_val'
    label_file = 'data/extract_number_to_tag_val.txt'
    val_data, val_labels = load_data(images_base, label_file)

    print(val_data.shape)
    print(val_labels.shape)

    result = model.predict(val_data)
    print(result)
    print("precision score: ", precision_score(val_labels, result, average='micro'))
    print("recall score: ", recall_score(val_labels, result, average='micro'))
