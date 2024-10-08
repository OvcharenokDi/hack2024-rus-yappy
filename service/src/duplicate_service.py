from repository import mark_duplicate,mark_duplicate_test, mark_hard, get_by_id, save_weight
from faiss_search import search
from similar_audio import compare_audio
from similar2 import compare_videos

## Интерфейс анализа дубликатов
def analyze(file_id):
    file_path = "/home/user1/hack2024-rus-yappy/service/temp/files/" + file_id + ".mp4"

    model_id, w = search(file_path)
    search_file_path = "/home/user1/hack2024-rus-yappy/service/temp/files/" + model_id + ".mp4"
    print(model_id)
    print(w)

    model_d = get_by_id(model_id)
    file_d = get_by_id(file_id)

    if w <= 100 and file_d.created > model_d.created:
        original_id = model_id
        duplicte_id = file_id
        original_time = 0
        duplicte_time = 0
        mark_duplicate(file_id, True, original_id)
        ##mark_hard(file_id,)
    elif 100 < w <= 135 and float(
            compare_videos(file_path, search_file_path)) > 0.93 and file_d.created > model_d.created:
        original_id = model_id
        duplicte_id = file_id
        original_time = 0
        duplicte_time = 0
        mark_duplicate(file_id, True, original_id)
        ##mark_hard(file_id,)
    else:
        original_id = file_id
        original_time = None
        duplicte_id = None
        duplicte_time = None


    save_weight(file_id, model_id, str(w), None)

    return {"origin":{"id": original_id, "time" : original_time},"dupliacte": {"id": duplicte_id, "time" : duplicte_time}}