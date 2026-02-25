from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .schemas import TaskStatusResponse, TaskSubmitResponse
from .security import verify_token
from .task_service import service

app = FastAPI(title="BraTS MRI Segmentation Service", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path("system/frontend")
if frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")


def _modality_from_name(name: str) -> str:
    n = name.lower()
    if "flair" in n:
        return "flair"
    if "t1ce" in n or "t1c" in n:
        return "t1ce"
    if "_t1" in n or n.endswith("t1.nii"):
        return "t1"
    if "t2" in n:
        return "t2"
    raise ValueError(f"Cannot infer modality from filename: {name}")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/tasks", response_model=TaskSubmitResponse, dependencies=[Depends(verify_token)])
async def submit_task(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    case_id: str = Form(default="anonymous_case"),
    backend: str = Form(default=settings.default_backend),
):
    if backend not in {"pytorch", "onnx", "tensorrt"}:
        raise HTTPException(status_code=400, detail="backend must be one of pytorch/onnx/tensorrt")

    if len(files) < 4:
        raise HTTPException(status_code=400, detail="Please upload 4 modality .nii files")

    state = service.create_task(case_id=case_id)
    case_dir = settings.upload_root / state.task_id
    case_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    for f in files:
        if not f.filename.endswith(".nii"):
            continue
        mod = _modality_from_name(f.filename)
        seen.add(mod)
        dst = case_dir / f"{state.task_id}_{mod}.nii"
        with dst.open("wb") as out:
            shutil.copyfileobj(f.file, out)

    required = {"flair", "t1", "t1ce", "t2"}
    if seen != required:
        raise HTTPException(status_code=400, detail=f"Uploaded modalities incomplete. got={sorted(seen)}")

    background_tasks.add_task(service.run, state.task_id, str(case_dir), backend)
    return TaskSubmitResponse(task_id=state.task_id, status=state.status)


@app.get("/api/tasks/{task_id}", response_model=TaskStatusResponse, dependencies=[Depends(verify_token)])
def get_task(task_id: str):
    try:
        st = service.get(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(**st.__dict__)


@app.get("/api/tasks/{task_id}/seg", dependencies=[Depends(verify_token)])
def download_seg(task_id: str):
    st = service.get(task_id)
    if st.status != "done" or not st.result_seg_nifti:
        raise HTTPException(status_code=400, detail="Task not finished")
    return FileResponse(st.result_seg_nifti, filename=f"{task_id}_seg.nii.gz")


@app.get("/api/tasks/{task_id}/report", dependencies=[Depends(verify_token)])
def download_report(task_id: str):
    st = service.get(task_id)
    if st.status != "done" or not st.report_pdf:
        raise HTTPException(status_code=400, detail="Task not finished")
    return FileResponse(st.report_pdf, filename=f"{task_id}_report.pdf")


@app.get("/api/tasks/{task_id}/preview/{view}", dependencies=[Depends(verify_token)])
def preview(task_id: str, view: str):
    allowed = {
        "axial",
        "sagittal",
        "coronal",
        "axial_base",
        "sagittal_base",
        "coronal_base",
        "axial_mask",
        "sagittal_mask",
        "coronal_mask",
    }
    if view not in allowed:
        raise HTTPException(status_code=400, detail="Invalid view")
    path = settings.result_root / task_id / f"{view}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(str(path), media_type="image/png")
