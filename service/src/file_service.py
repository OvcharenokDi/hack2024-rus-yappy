from pathlib import Path

import requests
import os
from urllib.parse import urlparse
from fastapi.responses import StreamingResponse
import aiofiles

from duplicate_service import analyze
from service.src.repository import get_duplicate_list_for_download, mark_download

CHUNK_SIZE = 1024 * 1024

def check_link(link):
    name = get_name(link)
    file_path = "../temp/files/" + name

    with open(file_path, 'wb') as out_file:
        content = requests.get(link, stream=True).content
        out_file.write(content)

    info = analyze(id, file_path)

    if info["dupliacte"]["id"] is not None:
        return {"is_duplicate": True, "duplicate_for": info["origin"]["id"]}
    else:
        return {"is_duplicate": False, "duplicate_for": None}

def load(link):
    name = get_name(link)
    file_path = "../temp/files/" + name
    id = get_id(name)

    with open(file_path, 'wb') as out_file:
        content = requests.get(link, stream=True).content
        out_file.write(content)

    return analyze(id, file_path)

def download(id):
    async def iterfile():
       async with aiofiles.open("../temp/files/" + id + ".mp4", 'rb') as f:
            while chunk := await f.read(CHUNK_SIZE):
                yield chunk

    headers = {'Content-Disposition': 'attachment; filename="' + id + '.mp4"'}
    return StreamingResponse(iterfile(), headers=headers, media_type='application/octet-stream')


def load_train():
    while True:
        list = get_duplicate_list_for_download()
        if not list:
            return

        for i in list:
            try:
                if is_exist(i.uuid):
                    print("File exist: " + i.link)
                else:
                    load(i.link)

                mark_download(i.uuid)
            except Exception as e:
                print("error in loading", i.link)

def get_name(link):
    a = urlparse(link)
    print(a.path)
    return os.path.basename(a.path)

def get_id(name):
    return name.split(".")[0]

def is_exist(id):
    my_file = Path("../temp/files/" + id + ".mp4")
    return my_file.is_file()
