import csv

import uvicorn

from file_service import check_link, load, download
from repository import init_db, create
from fastapi import FastAPI, Body

init_db()
app = FastAPI()

if __name__ == '__main__':
    uvicorn.run('main:app', workers=1, host="0.0.0.0", port=9000)

@app.post("/check-video-duplicate")
async def post_check(data = Body()):
    return check_link(data["link"])


@app.post("/init_db")
async def init_db():
    with open('train.csv', 'r', encoding='utf-8-sig', newline='') as file:
        reader = csv.reader(file)
        header = list(next(reader))
        for items in reader:
            create(items)
            print(items)


@app.post("/load")
async def post_load(data = Body()):
    return load(data["link"])

@app.get("/download/{id}")
async def post_download(id):
    return download(id)