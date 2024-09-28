from pathlib import Path

import requests
import os
from urllib.parse import urlparse
from fastapi.responses import StreamingResponse
import aiofiles

from duplicate_service import analyze
from repository import get_duplicate_list_for_download, mark_download, get_all_train

CHUNK_SIZE = 1024 * 1024


def check_link(link):
    name = get_name(link)
    id = get_id(name)

    if not is_exist(id):
        load_file(link)

    info = analyze(id)

    if info["dupliacte"]["id"] is not None:
        return {"is_duplicate": True, "duplicate_for": info["origin"]["id"]}
    else:
        return {"is_duplicate": False, "duplicate_for": None}


def analysis(link):
    name = get_name(link)
    id = get_id(name)

    if not is_exist(id):
        load_file(link)

    return analyze(id)


def load_file(link):
    name = get_name(link)
    file_path = "/home/user1/hack2024-rus-yappy/service/temp/files/" + name

    with open(file_path, 'wb') as out_file:
        content = requests.get(link, stream=True).content
        out_file.write(content)

    return file_path


def download(id):
    if not is_exist(id):
        load_file("https://s3.ritm.media/yappy-db-duplicates/" + id + ".mp4")

    async def iterfile():
        async with aiofiles.open("/home/user1/hack2024-rus-yappy/service/temp/files/" + id + ".mp4", 'rb') as f:
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
                    load_file(i.link)

                mark_download(i.uuid)
            except Exception as e:
                print("error in loading", i.link)


def all_analysis():
    list = get_all_train()
    for i in list:
        try:
            if is_exist(i.uuid):
                print("File exist: " + i.link)
            else:
                load_file(i.link)
            analyze(i.uuid)
        except Exception as e:
            print("error in loading", i.link)


def create_csv():
    list = get_all_train()
    with open('/home/user1/hack2024-rus-yappy/service/temp/submission.csv', 'w') as f:
        f.write('created,uuid,link,is_duplicate,duplicate_for,is_hard')
        f.write('\n')
        for i in list:
            f.write(i.created.strftime("%Y-%m-%d %H:%M:%S") + "," + i.uuid + "," + i.link + "," + str(i.is_duplicate) + "," + i.duplicate_for + "," + str(i.is_hard))
            f.write('\n')


def get_name(link):
    a = urlparse(link)
    print(a.path)
    return os.path.basename(a.path)


def get_id(name):
    return name.split(".")[0]


def is_exist(id):
    my_file = Path("/home/user1/hack2024-rus-yappy/service/temp/files/" + id + ".mp4")
    return my_file.is_file()
