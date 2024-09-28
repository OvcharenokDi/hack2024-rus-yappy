import glob
import math
import os
import pickle
import random
import shutil
import uuid

import cv2
import faiss
import imagehash
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from skimage.feature import local_binary_pattern
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms

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


def crop_resistant_hash(img, hash_func=imagehash.whash, hash_size=8, mode='haar', multires=True, grid_size=3,
                        overlap=0.5):
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
                h = hash_func(cropped_img, mode=mode, hash_size=hash_size)
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
        # whash = imagehash.whash(img, image_scale=64, hash_size=32, mode='haar')
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

        # Отражение по горизонтали
        mirror_hash = imagehash.phash(ImageOps.mirror(img))

        flip_hash = imagehash.phash(ImageOps.flip(img))

        rotate90_hash = imagehash.phash(img.rotate(90, expand=True))

        # Поворот на 180 градусов
        rotate180_hash = imagehash.phash(img.rotate(180, expand=True))

        # Поворот на 270 градусов
        rotate270_hash = imagehash.phash(img.rotate(270, expand=True))

        # Негатив
        inverted_hash = imagehash.phash(ImageOps.invert(img))

        # Преобразование хешей в плоские массивы
        features = [
            # cr_hash,
            whash.hash.flatten(),
            phash.hash.flatten(),
            colorhash.hash.flatten(),
            # color_hist.flatten(),
            # lbp_hist.flatten(),
            # orb_features.flatten(),
            mirror_hash.hash.flatten(),
            flip_hash.hash.flatten(),
            rotate90_hash.hash.flatten(),
            rotate180_hash.hash.flatten(),
            rotate270_hash.hash.flatten(),
            inverted_hash.hash.flatten(),
            neural
        ]

        return features
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return None


def process_image(image_path):
    hashes = calculate_hashes_and_features(image_path)
    return np.concatenate(hashes)


def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    return index


def load_dataset(dataset_path):
    dataset = np.load(dataset_path)
    return dataset


def search_similar_images(query_image_path, index, dataset, paths, top_k=5):
    # Обработка запроса
    query_vector = process_image(query_image_path)
    query_vector = np.expand_dims(query_vector, axis=0).astype('float32')

    with open('/home/user1/faiss/minmax_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

        if query_vector.shape[1] != scaler.n_features_in_:
            raise ValueError(f"Ожидалось {scaler.n_features_in_} признаков, получено {query_vector.shape[1]}.")

        query_vector = scaler.transform(query_vector)

        # Поиск в индексе
        distances, indices = index.search(query_vector, top_k)

        # Получение путей к похожим изображениям (предполагается, что у вас есть список путей)
        similar_images = []
        for idx in indices[0]:
            similar_image_path = paths[idx]
            # Здесь предполагается, что dataset содержит пути к изображениям
            similar_images.append(similar_image_path)

        return similar_images, distances[0]


def search_similar_images_by_uuid(query_image):
    uuid = query_image.split('/')[-1].split('_')[0]

    # Выполнение поиска
    similar_images, distances = search_similar_images(query_image, index, dataset, paths, top_k=500)

    top_5_results = []

    # Вывод результатов
    print(f"Похожие изображения для {query_image}:")
    count = 0
    for img, dist in zip(similar_images, distances):
        if not img.startswith(f"./dataset_new2/{uuid}"):
            print(f"Путь: {img}, Расстояние: {dist}")
            top_5_results.append({"img": img, "dist": dist})
            if count == 5:
                return top_5_results
            count += 1


def frame_difference(frame1, frame2, threshold=25):
    # Преобразование кадров в grayscale для упрощения обработки
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Вычисление абсолютной разницы между кадрами
    diff = cv2.absdiff(gray1, gray2)

    # Применение порогового значения
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Вычисление процентного соотношения измененных пикселей
    non_zero_count = cv2.countNonZero(thresh)
    total_pixels = thresh.size
    percentage = (non_zero_count / total_pixels) * 100

    return percentage


def extract_frames(video_path, output_dir):
    # Получаем имя видео без расширения
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Не удалось открыть видеофайл {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    # print(f"Кадровая частота (FPS): {fps}")

    ret, prev_frame = cap.read()
    frame_count = 0
    saved_count = 0

    while ret:
        ret, current_frame = cap.read()
        if not ret:
            break

        # Получение текущего номера кадра
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

        # Вычисление времени в секундах
        time_seconds = frame_number / fps

        if saved_count == 0:
            # Создаем имя файла для кадра
            frame_filename = f"{video_name}_{time_seconds}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)

            # Сохраняем кадр
            cv2.imwrite(frame_path, current_frame)
            saved_count += 1
        else:
            difference = frame_difference(prev_frame, current_frame)
            if difference >= 7:
                # Создаем имя файла для кадра
                frame_filename = f"{video_name}_{time_seconds}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)

                # Сохраняем кадр
                cv2.imwrite(frame_path, current_frame)
                saved_count += 1
                # print(f"Сохранен кадр {frame_count} с разницей {difference:.2f}%")

        prev_frame = current_frame
        frame_count += 1

    cap.release()
    print(f"Извлечено {saved_count} кадров из {video_path} с разницей 7% и более.")


def get_frame_histogram(frame):
    """
    Вычисляет гистограмму цвета для кадра.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Используем гистограмму для каналов H и S
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def are_histograms_similar(hist1, hist2, threshold=0.7):
    """
    Сравнивает две гистограммы с использованием коэффициента корреляции.
    """
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return correlation > threshold


def extract_diverse_frames(video_path, max_frames=30):
    """
    Извлекает максимальное количество (не более max_frames) наиболее разнообразных кадров из видео.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видео.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    selected_frames = []
    histograms = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    interval = max(1, int(frame_count / (max_frames * 2)))  # Интервалы для выбора кадров

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Получение текущего номера кадра
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

        # Вычисление времени в секундах
        time_seconds = frame_number / fps

        if current_frame % interval == 0:
            hist = get_frame_histogram(frame)

            # Проверяем, похож ли текущий кадр на уже выбранные
            similar = False
            for existing_hist in histograms:
                if are_histograms_similar(hist, existing_hist):
                    similar = True
                    break

            if not similar:
                selected_frames.append({"video_name": video_name, "frame": frame, "time_seconds": time_seconds})
                histograms.append(hist)
                if len(selected_frames) >= max_frames:
                    break

        current_frame += 1

    cap.release()
    return selected_frames


def select_unique_frames(video_path, max_frames=30, hash_size=8, frame_interval=10):
    """
    Извлекает уникальные кадры из видео.

    :param video_path: Путь к видеофайлу.
    :param max_frames: Максимальное количество уникальных кадров.
    :param hash_size: Размер хэша для сравнения (чем больше, тем точнее).
    :param frame_interval: Интервал между кадрами для выборки.
    :return: Список уникальных кадров в формате OpenCV.
    """
    # Получаем имя видео без расширения
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка открытия видео файла.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)

    unique_frames = []
    hashes = set()
    frame_count = 0

    while cap.isOpened() and len(unique_frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Получение текущего номера кадра
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

        # Вычисление времени в секундах
        time_seconds = frame_number / fps

        if frame_count % frame_interval == 0:
            # Преобразуем кадр из BGR (OpenCV) в RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Вычисляем хэш кадра
            frame_hash = imagehash.average_hash(pil_image, hash_size=hash_size)

            if frame_hash not in hashes:
                hashes.add(frame_hash)
                unique_frames.append({"video_name": video_name, "frame": frame, "time_seconds": time_seconds})

        frame_count += 1

    cap.release()
    return unique_frames


def remove_dir(path):
    try:
        shutil.rmtree(path)
        print(f"Директория '{path}' и все её содержимое успешно удалены.")
    except Exception as e:
        print(f"Ошибка при удалении директории: {e}")


def extract_frames(video_path, max_frames=100, step=30):
    """
    Извлекает кадры из видео с заданным шагом.

    :param video_path: Путь к видеофайлу.
    :param max_frames: Максимальное количество извлечённых кадров.
    :param step: Шаг извлечения кадров (количество пропускаемых кадров).
    :return: Список извлечённых кадров.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            # Конвертируем цветовое пространство из BGR в RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        count += 1

    cap.release()
    return frames


def compute_features(frames):
    """
    Вычисляет вектор признаков для каждого кадра.

    :param frames: Список кадров.
    :return: Массив признаков.
    """
    features = []
    for frame in frames:
        # Простой пример: усреднённые цветовые значения
        feature = frame.mean(axis=(0, 1))
        features.append(feature)
    return np.array(features)


def select_representative_frames(frames, features, num_frames=5):
    """
    Выбирает наиболее репрезентативные кадры на основе средней близости.

    :param frames: Список кадров.
    :param features: Массив признаков.
    :param num_frames: Количество кадров для выбора.
    :return: Список выбранных кадров.
    """
    similarity_matrix = cosine_similarity(features)
    # Вычисляем среднее сходство каждого кадра с другими
    mean_similarities = similarity_matrix.mean(axis=1)
    # Получаем индексы кадров с наивысшей средней сходностью
    selected_indices = np.argsort(mean_similarities)[-num_frames:]
    # Сортируем выбранные индексы по порядку появления в видео
    selected_indices = sorted(selected_indices)
    return [frames[i] for i in selected_indices]


def save_frames(frames, output_folder):
    for idx, frame in enumerate(frames):
        filename = f"file_{idx + 1}.jpg"
        frame_path = os.path.join(output_folder, filename)
        plt.imsave(frame_path, frame)


def top_5_videos(video_path):
    temp_dir = f'/home/user1/hack2024-rus-yappy/service/temp/{uuid.uuid4()}'
    os.mkdir(temp_dir)

    frames = extract_frames(video_path, max_frames=60, step=30)
    print(f"Извлечено {len(frames)} кадров из видео.")

    features = compute_features(frames)
    selected_frames = select_representative_frames(frames, features, num_frames=5)

    # unique_frames = extract_diverse_frames(video_path)
    # save_frames(unique_frames, temp_dir)
    save_frames(selected_frames, temp_dir)

    # extract_frames(video_path=video_path, output_dir=temp_dir)
    image_paths = glob.glob(os.path.join(temp_dir, '**', '*.jpg'), recursive=True)

    sample_size = max(1, int(len(image_paths) * 1))
    sampled_image_paths = random.sample(image_paths, sample_size) if len(image_paths) >= sample_size else image_paths

    uuid_distances = {}

    for query_image in sampled_image_paths:
        top_5_video = search_similar_images_by_uuid(query_image)
        # Извлекаем img_uuid из пути изображения
        img_uuid = top_5_video[0]['img'].split('/')[-1].split('_')[0]
        current_dist = top_5_video[0]['dist']

        # Если img_uuid еще не в словаре, добавляем его с текущим расстоянием
        if img_uuid not in uuid_distances:
            uuid_distances[img_uuid] = current_dist
        else:
            # Сравниваем и сохраняем минимальное расстояние
            if current_dist < uuid_distances[img_uuid]:
                uuid_distances[img_uuid] = current_dist

    # Сортируем словарь по значению расстояния в порядке возрастания
    sorted_uuid_distances = sorted(uuid_distances.items(), key=lambda item: item[1])

    # Получаем первый элемент из отсортированного списка
    first_uuid, first_distance = sorted_uuid_distances[0]
    print(f"UUID с наименьшим расстоянием: {first_uuid}, Расстояние: {first_distance}")
    return first_uuid, first_distance


def search(video_path):
    # Пути к индексам и данным
    index_path = '/home/user1/faiss/image_index.faiss'
    dataset_path = '/home/user1/faiss/image_dataset.npy'
    path_path = '/home/user1/faiss/path_dataset.npy'

    # Загрузка индекса и набора данных
    index = load_faiss_index(index_path)
    dataset = load_dataset(dataset_path)
    paths = load_dataset(path_path)
    return top_5_videos(video_path)
