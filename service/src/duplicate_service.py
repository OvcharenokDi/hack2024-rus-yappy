from repository import mark_duplicate, mark_hard, get_by_id


def analyze(file_id):
    file_path = "../temp/files/" + file_id + ".mp4"

    duplicate = get_by_id(file_id)
    if duplicate.is_duplicate == True:
        original_id = duplicate.duplicate_for
        original_time = 0
        duplicte_id = file_id
        duplicte_time = 0
    else:
        original_id = file_id
        original_time = None
        duplicte_id = None
        duplicte_time = None

    return {"origin":{"id": original_id, "time" : original_time},"dupliacte": {"id": duplicte_id, "time" : duplicte_time}}