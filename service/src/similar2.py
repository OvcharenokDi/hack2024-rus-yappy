import torch
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
from PIL import Image
import cv2
import numpy as np
import requests
from io import BytesIO
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint").to(device)

def load_video_frames(video_path, max_frames=10):
    """
    Load frames from a video file, sampling up to max_frames evenly spaced.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames


def extract_features(frames, processor, model, device):
    """
    Extract keypoints, scores, and descriptors from a list of PIL Images.
    """
    inputs = processor(images=frames, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    keypoints_list = []
    descriptors_list = []
    for i in range(len(frames)):
        mask = outputs.mask[i]
        indices = torch.nonzero(mask).squeeze()
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        keypoints = outputs.keypoints[i][indices].cpu()
        descriptors = outputs.descriptors[i][indices].cpu()
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
    return keypoints_list, descriptors_list


def compute_similarity(descriptors1, descriptors2):
    """
    Compute similarity between two sets of descriptors using cosine similarity.
    """
    # Normalize descriptors
    descriptors1 = torch.nn.functional.normalize(torch.cat(descriptors1, dim=0), p=2, dim=1)
    descriptors2 = torch.nn.functional.normalize(torch.cat(descriptors2, dim=0), p=2, dim=1)
    # Compute cosine similarity matrix
    similarity_matrix = torch.mm(descriptors1, descriptors2.t())
    # Take the maximum similarity for each descriptor in descriptors1
    max_similarities, _ = similarity_matrix.max(dim=1)
    # Average the maximum similarities
    similarity_score = max_similarities.mean().item()
    # Normalize to [0,1] if not already
    similarity_score = (similarity_score + 1) / 2  # cosine similarity ranges from -1 to 1
    return similarity_score


def compare_videos(video_path_1, video_path_2, max_frames=20):
    start_time = time.time()

    # Load frames from both videos
    frames1 = load_video_frames(video_path_1, max_frames)
    frames2 = load_video_frames(video_path_2, max_frames)

    # Extract features
    keypoints1, descriptors1 = extract_features(frames1, processor, model, device)
    keypoints2, descriptors2 = extract_features(frames2, processor, model, device)

    # Compute similarity
    similarity = compute_similarity(descriptors1, descriptors2)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Similarity Score: {similarity:.4f}")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    if elapsed_time > 3:
        print("Warning: Execution time exceeded 3 seconds.")
    return similarity

