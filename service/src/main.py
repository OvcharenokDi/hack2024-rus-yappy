import csv

import uvicorn
from pydantic import BaseModel

from file_service import check_link, analysis, download, load_train, all_analysis, create_csv
from repository import init_db, create, add
from fastapi import FastAPI, Body, HTTPException

init_db()
app = FastAPI()

if __name__ == '__main__':
    uvicorn.run('main:app', workers=2, host="0.0.0.0", port=9001)


class LinkDto(BaseModel):
    link: str = None

@app.post("/check-video-duplicate")
async def post_check(linkDto: LinkDto = None):
    if linkDto is None or linkDto.link is None:
        raise HTTPException(status_code=400)
    return check_link(linkDto.link)

@app.post("/analysis")
async def post_load(data = Body()):
    return analysis(data["link"])

@app.get("/download/{id}")
async def post_download(id):
    return download(id)

@app.post("/technical/load/train/{key}")
async def post_load_train(key):
    check_key(key)
    return load_train()

@app.post("/technical/init_db/{key}")
async def init_db(key):
    check_key(key)
    with open('/home/user1/hack2024-rus-yappy/service/temp/train.csv', 'r', encoding='utf-8-sig', newline='') as file:
        reader = csv.reader(file)
        header = list(next(reader))
        for items in reader:
            create(items)
            print(items)

@app.post("/technical/train_db/{key}")
async def init_db(key):
    check_key(key)
    with open('/home/user1/hack2024-rus-yappy/service/temp/train.csv', 'r', encoding='utf-8-sig', newline='') as file:
        reader = csv.reader(file)
        header = list(next(reader))
        for items in reader:
            add(items)
            print(items)

@app.post("/technical/analysis/{key}")
async def init_db(key):
    check_key(key)
    all_analysis()


@app.post("/technical/submission/{key}")
async def init_db(key):
    check_key(key)
    create_csv()

def check_key(key):
    if key != '9991':
        raise Exception('403 Forbidden')