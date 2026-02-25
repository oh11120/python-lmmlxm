from __future__ import annotations

from fastapi import Header, HTTPException, Query

from .config import settings


def verify_token(x_api_token: str = Header(default=""), token: str = Query(default="")) -> None:
    candidate = x_api_token or token
    if candidate != settings.api_token:
        raise HTTPException(status_code=401, detail="Invalid API token")
