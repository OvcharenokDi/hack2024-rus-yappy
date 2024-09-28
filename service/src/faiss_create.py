import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import faiss
import imagehash
import numpy as np
import torch
from PIL import Image
from skimage.feature import local_binary_pattern
from torchvision import models, transforms
import math
from sklearn.preprocessing import MinMaxScaler
import pickle

model = models.resnet50(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def neural_hash(img):
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    return output.numpy().flatten()


def crop_resistant_hash(img, hash_func=imagehash.whash, hash_size=8, mode='haar', multires=True, grid_size=3, overlap=0.5):
    if multires:
        hashes = []
        width, height = img.size
        grid_width = math.ceil(width / grid_size)
        grid_height = math.ceil(height / grid_size)

        for i in range(grid_size):
            for j in range(grid_size):
                left = int(j * grid_width * (1 - overlap))
                upper = int(i * grid_height * (1 - overlap))
                right = min(left + grid_width, width)
                lower = min(upper + grid_height, height)
                cropped_img = img.crop((left, upper, right, lower))
                h = hash_func(cropped_img, hash_size=hash_size, mode=mode)
                hashes.append(h.hash.flatten())

        combined_hash = np.concatenate(hashes)
        return combined_hash
    else:
        return hash_func(img, hash_size=hash_size).hash.flatten()


def calculate_hashes_and_features(image_path):
    try:
        # Открытие изображения и конвертация в RGB
        img = Image.open(image_path).convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Вычисление хешей
        phash = imagehash.phash(img, hash_size=8, highfreq_factor=4)
        whash = imagehash.whash(img, hash_size=8, mode='db2')
        colorhash = imagehash.colorhash(img, binbits=8)
        cr_hash = crop_resistant_hash(img, hash_func=imagehash.whash, hash_size=8, mode='db2', multires=True)

        # Добавление цветовой гистограммы
        color_hist = cv2.calcHist([img_cv], [0, 1, 2], None, [8, 8, 8],
                                  [0, 256, 0, 256, 0, 256])
        color_hist = cv2.normalize(color_hist, color_hist).flatten()

        # print(f"HOG {image_path}")
        #
        # # # Добавление признаков HOG
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # hog_features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
        #                               cells_per_block=(2, 2), block_norm='L2-Hys',
        #                               visualize=True, feature_vector=True)

        # Добавление Local Binary Patterns (LBP)
        lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
        (lbp_hist, _) = np.histogram(lbp.ravel(),
                                     bins=np.arange(0, 24 + 3),
                                     range=(0, 24 + 2))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-7)

        # Добавление ключевых точек и дескрипторов (ORB)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is not None:
            # Усреднение дескрипторов или использование другого подхода
            orb_features = descriptors.mean(axis=0)
        else:
            orb_features = np.zeros(orb.descriptorSize())

        neural = neural_hash(img)

        # Преобразование хешей в плоские массивы
        features = [
            # cr_hash,
            whash.hash.flatten(),
            phash.hash.flatten(),
            # colorhash.hash.flatten(),
            # color_hist.flatten(),
            # lbp_hist.flatten(),
            # orb_features.flatten(),
            neural
        ]

        return features
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return None


def process_image(image_path):
    h = calculate_hashes_and_features(image_path)
    if h:
        features = np.concatenate(h)
        return features, image_path
    return None


def process_directory(directory_path):
    image_paths = glob.glob(os.path.join(directory_path, '**', '*.jpg'), recursive=True)
    hashes = []
    paths = []
    count = 1
    max_threads = 40

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Создаем генератор задач
        future_to_image = {executor.submit(process_image, img_path): img_path for img_path in image_paths}

        for future in as_completed(future_to_image):
            result = future.result()
            if result:
                combined_hash, image_path = result
                hashes.append(combined_hash)
                paths.append(image_path)
                print(count, len(image_paths) / count, image_path)
                count += 1

    return np.array(hashes), paths


def create_faiss_index(data):
    dimension = data.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Используем L2 расстояние
    index.add(data.astype('float32'))
    return index


# Пример использования
if __name__ == "__main__":
    directory = './dataset_new2'
    dataset, paths = process_directory(directory)

    # Инициализация скалера, если не передан
    scaler = MinMaxScaler()

    # Применение нормализации
    dataset = scaler.fit_transform(dataset)

    with open('../temp/newmodel/minmax_scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    np.save('../temp/newmodel/image_dataset.npy', dataset)
    np.save('../temp/newmodel/path_dataset.npy', paths)
    print(f"Количество обработанных изображений: {len(dataset)}")

    # Создание индекса FAISS
    index = create_faiss_index(dataset)
    print("Индекс FAISS создан и обучен.")

    # Сохранение индекса и данных, если необходимо
    faiss.write_index(index, '../temp/newmodel/image_index.faiss')
