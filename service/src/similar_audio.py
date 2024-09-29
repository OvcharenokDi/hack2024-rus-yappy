import ffmpeg
import torch
import torchaudio
import time


def extract_audio(video_path, audio_path, duration=10):
    """Извлекает аудио из видеофайла с ограничением по длительности."""
    (
        ffmpeg
        .input(video_path, t=duration)
        .output(audio_path, ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )


def load_audio(audio_path, device):
    """Загружает аудиофайл и переносит его на устройство (CPU или GPU)."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    waveform = waveform.to(device)
    return waveform


def extract_features(waveform, device):
    """Извлекает спектрограмму из аудиофайла."""
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128).to(device)
    with torch.no_grad():
        features = transform(waveform)
        features = torch.mean(features, dim=2)  # Усреднение по времени
        features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features


def compute_cosine_similarity(features1, features2):
    """Вычисляет косинусное сходство между двумя векторами признаков."""
    similarity = torch.nn.functional.cosine_similarity(features1, features2)
    return similarity.item()


def compare_audio_similarity(video1, video2):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Извлечение аудио
    extract_audio(video1, 'audio1.wav')
    extract_audio(video2, 'audio2.wav')

    # Загрузка аудио
    waveform1 = load_audio('audio1.wav', device)
    waveform2 = load_audio('audio2.wav', device)

    # Извлечение признаков
    features1 = extract_features(waveform1, device)
    features2 = extract_features(waveform2, device)

    # Вычисление сходства
    similarity = compute_cosine_similarity(features1, features2)

    # Нормализация метрики (значение от 0 до 1)
    similarity_normalized = (similarity + 1) / 2

    end_time = time.time()
    execution_time = end_time - start_time
    if execution_time > 3:
        print(f"Предупреждение: Время выполнения превысило 3 секунды ({execution_time:.2f} секунд).")

    return similarity_normalized


def compare_audio(video_path_1, video_path_2):
    similarity_score = compare_audio_similarity(video_path_1, video_path_2)
    print(f"Похожесть аудио: {similarity_score:.4f}")
    return similarity_score
