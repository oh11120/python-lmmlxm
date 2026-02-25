from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class TaskSubmitResponse(BaseModel):
    task_id: str
    status: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    message: str
    progress: float
    result_seg_nifti: Optional[str] = None
    report_pdf: Optional[str] = None


class InferenceOptions(BaseModel):
    backend: str = "pytorch"
    threshold: float = 0.5
