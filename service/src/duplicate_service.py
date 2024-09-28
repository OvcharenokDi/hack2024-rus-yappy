from repository import mark_duplicate, mark_hard, get_by_id, save_weight
from faiss_search import search


def analyze(file_id):
    file_path = "/home/user1/hack2024-rus-yappy/service/temp/files/" + file_id + ".mp4"

    origin_id, w = search(file_path)

    print(origin_id)
    print(w)
    save_weight(file_id, origin_id, str(w))
    if w > 0.5:
        original_id = origin_id
        duplicte_id = file_id
        original_time = 0
        duplicte_time = 0
        #mark_duplicate(file_id, True, original_id)
        ##mark_hard(file_id,)
    else:
        original_id = file_id
        original_time = None
        duplicte_id = None
        duplicte_time = None

    return {"origin":{"id": original_id, "time" : original_time},"dupliacte": {"id": duplicte_id, "time" : duplicte_time}}