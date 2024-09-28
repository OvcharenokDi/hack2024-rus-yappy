from repository import mark_duplicate, mark_hard, get_by_id
from faiss_search import search


def analyze(file_id):
    file_path = "../temp/files/" + file_id + ".mp4"

    info = search(file_path)

    if info[1] > 0.5:
        original_id = info[0]
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

    return {"origin":{"id": original_id, "time" : original_time},"dupliacte": {"id": duplicte_id, "time" : duplicte_time}}