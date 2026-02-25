from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

from .config import settings
from .logging_utils import get_backend_logger
from .report import generate_pdf_report
from .visualize import generate_preview_images


@dataclass
class TaskState:
    task_id: str
    status: str
    message: str
    progress: float
    case_id: str
    result_seg_nifti: Optional[str] = None
    report_pdf: Optional[str] = None


class TaskService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: Dict[str, TaskState] = {}
        self._logger = get_backend_logger()

    def _state_path(self, task_id: str) -> Path:
        return settings.task_root / f"{task_id}.json"

    def _audit_log(self, task_id: str, event: str, message: str) -> None:
        log_path = settings.task_root / "audit.log"
        line = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "task_id": task_id,
            "event": event,
            "message": message,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def _persist(self, state: TaskState) -> None:
        self._state_path(state.task_id).write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")

    def create_task(self, case_id: str) -> TaskState:
        task_id = uuid.uuid4().hex
        state = TaskState(task_id=task_id, status="queued", message="Task created", progress=0.0, case_id=case_id)
        with self._lock:
            self._states[task_id] = state
        self._persist(state)
        self._audit_log(task_id, "create", f"case_id={case_id}")
        self._logger.info("task=%s create case_id=%s", task_id, case_id)
        return state

    def update(self, task_id: str, **kwargs) -> TaskState:
        with self._lock:
            st = self._states[task_id]
            for k, v in kwargs.items():
                setattr(st, k, v)
            self._states[task_id] = st
        self._persist(st)
        if "status" in kwargs or "message" in kwargs:
            self._audit_log(task_id, "update", f"status={st.status}; message={st.message}")
            self._logger.info("task=%s status=%s message=%s progress=%.2f", task_id, st.status, st.message, st.progress)
        return st

    def get(self, task_id: str) -> TaskState:
        with self._lock:
            st = self._states.get(task_id)
        if st:
            return st

        path = self._state_path(task_id)
        if not path.exists():
            raise KeyError(task_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        st = TaskState(**payload)
        with self._lock:
            self._states[task_id] = st
        return st

    def run(self, task_id: str, case_dir: str, backend: str) -> None:
        try:
            from .inference import InferenceEngine

            self._logger.info("task=%s run start backend=%s case_dir=%s", task_id, backend, case_dir)
            self.update(task_id, status="running", message="Loading model", progress=0.1)
            engine = InferenceEngine(
                backend=backend,
                model_name=settings.model_name,
                ckpt_path=settings.model_ckpt,
                onnx_path=settings.onnx_path,
                trt_engine_path=settings.trt_engine_path,
            )

            out_seg = settings.result_root / f"{task_id}_seg.nii.gz"
            self.update(task_id, message="Running inference", progress=0.45)
            infer_out = engine.infer_case(case_dir=case_dir, output_seg_path=str(out_seg))

            flair_path = str(next(Path(case_dir).glob("*_flair.nii")))
            self.update(task_id, message="Rendering previews", progress=0.7)
            preview = generate_preview_images(flair_path=flair_path, seg_path=infer_out["seg_path"], out_dir=str(settings.result_root / task_id))

            report_path = settings.report_root / f"{task_id}.pdf"
            self.update(task_id, message="Generating PDF report", progress=0.85)
            generate_pdf_report(
                report_path=str(report_path),
                case_id=self.get(task_id).case_id,
                seg_path=infer_out["seg_path"],
                preview_images=preview,
                backend_name=backend,
            )

            self.update(
                task_id,
                status="done",
                message="Finished",
                progress=1.0,
                result_seg_nifti=str(out_seg),
                report_pdf=str(report_path),
            )
            self._logger.info("task=%s completed seg=%s report=%s", task_id, out_seg, report_path)
        except Exception as exc:
            self.update(task_id, status="failed", message=f"{type(exc).__name__}: {exc}", progress=1.0)
            self._logger.exception("task=%s failed", task_id)


service = TaskService()
