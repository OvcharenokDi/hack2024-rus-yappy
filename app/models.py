# generated by fastapi-codegen:
#   filename:  swagger.yaml
#   timestamp: 2024-09-28T10:36:04+00:00

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class VideoLinkRequest(BaseModel):
    link: Optional[str] = Field(
        None, description='ссылка на видео', example='https://example.com/video.mp4'
    )


class DuplicateFor(Enum):
    field_0003d59f_89cb_4c5c_9156_6c5bc07c6fad = '0003d59f-89cb-4c5c-9156-6c5bc07c6fad'
    field_000ab50a_e0bd_4577_9d21_f1f426144321 = '000ab50a-e0bd-4577-9d21-f1f426144321'


class VideoLinkResponse(BaseModel):
    is_duplicate: Optional[bool] = Field(
        None, description='признак дублирования', example=False
    )
    duplicate_for: Optional[DuplicateFor] = Field(
        None, description='идентификтаор видео в формате uuid4'
    )
