# Yappy hacks-ai


## Тестовый сервис:
Находится по адресу http://176.123.163.85:9001/check-video-duplicate

Ручки для фронта

/analysis - по ссылке от видео находит дубликат
```bash
   curl --location '176.123.163.85:9001/analysis' \
    --header 'Content-Type: application/json' \
    --data '{
    	"link": "https://s3.ritm.media/yappy-db-duplicates/d4341c53-cb65-4fcc-8882-fe32ff34aa34.mp4"
    }'
```

В ответе:
```json
   {
    "origin": {
        "id": "22d891cc-563a-48c9-9b6e-368829598e91",
        "time": 0
    },
    "dupliacte": {
        "id": "d4341c53-cb65-4fcc-8882-fe32ff34aa34",
        "time": 0
    }
}
```

/download - скачивает видео
```bash
   curl --location '176.123.163.85:9001/download/d9060c03-304c-45c8-b3e6-417caf07f7a7'
```

Для проверки
/check-video-duplicate - по ссылке от видео определяет оригинал/дубликат
```bash
   curl --location '176.123.163.85:9001/check-video-duplicate' \
    --header 'Content-Type: application/json' \
    --data '{
    	"link": "https://s3.ritm.media/yappy-db-duplicates/d4341c53-cb65-4fcc-8882-fe32ff34aa34.mp4"
    }'
```

В ответе:
```json
    {
        "is_duplicate": true,
        "duplicate_for": "22d891cc-563a-48c9-9b6e-368829598e91"
    }
```


## Описание файлов в проекте

- `create_dataset.ipynb` - ноутбук для создания датасета для faiss
- `faiss_create.ipynb` - ноутбук для создания индекса faiss
- `faiss_search.ipynb` - ноутбук для проверки индекса faiss
- `service` - проект для реализации ручек

