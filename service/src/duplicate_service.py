import uuid


def analyze(file_id, file_path):
    original_id = 'a5ff586a-e72c-4b1c-b3fe-53f1ad41efb7'
    original_time = 2

    duplicte_id = file_id
    duplicte_time = 3

    return {"origin":{"id": original_id, "time" : original_time},"dupliacte": {"id": duplicte_id, "time" : duplicte_time}}