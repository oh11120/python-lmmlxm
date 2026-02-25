const el = (id) => document.getElementById(id);

let activeTaskId = null;
let pollTimer = null;

const tokenEl = el("token");
const caseEl = el("caseId");
const backendEl = el("backend");
const filesEl = el("files");
const submitBtn = el("submitBtn");
const submitMsg = el("submitMsg");

const taskIdEl = el("taskId");
const taskStatusEl = el("taskStatus");
const taskProgressEl = el("taskProgress");
const taskMessageEl = el("taskMessage");

const alphaEl = el("alpha");
const alphaValueEl = el("alphaValue");
const downloadSegBtn = el("downloadSeg");
const downloadPdfBtn = el("downloadPdf");

function authHeaders() {
  return { "x-api-token": tokenEl.value.trim() };
}

function setStatus(data) {
  taskIdEl.textContent = data.task_id;
  taskStatusEl.textContent = data.status;
  taskProgressEl.textContent = `${Math.round((data.progress || 0) * 100)}%`;
  taskMessageEl.textContent = data.message || "-";

  if (data.status === "done") {
    downloadSegBtn.disabled = false;
    downloadPdfBtn.disabled = false;
    loadPreviews(data.task_id);
  }
}

function previewUrl(taskId, key) {
  const token = encodeURIComponent(tokenEl.value.trim());
  return `/api/tasks/${taskId}/preview/${key}?token=${token}&t=${Date.now()}`;
}

function loadPreviews(taskId) {
  ["axial", "sagittal", "coronal"].forEach((v) => {
    el(`${v}Base`).src = previewUrl(taskId, `${v}_base`);
    el(`${v}Mask`).src = previewUrl(taskId, `${v}_mask`);
  });
}

async function pollTask() {
  if (!activeTaskId) return;
  const res = await fetch(`/api/tasks/${activeTaskId}`, { headers: authHeaders() });
  if (!res.ok) {
    taskMessageEl.textContent = `查询失败: ${res.status}`;
    return;
  }
  const data = await res.json();
  setStatus(data);
  if (data.status === "done" || data.status === "failed") {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

submitBtn.addEventListener("click", async () => {
  try {
    submitMsg.textContent = "正在提交...";
    const files = filesEl.files;
    if (!files || files.length < 4) {
      submitMsg.textContent = "请至少上传4个nii文件";
      return;
    }

    const fd = new FormData();
    fd.append("case_id", caseEl.value || "demo_case");
    fd.append("backend", backendEl.value);
    Array.from(files).forEach((f) => fd.append("files", f));

    const res = await fetch("/api/tasks", {
      method: "POST",
      headers: authHeaders(),
      body: fd,
    });

    if (!res.ok) {
      submitMsg.textContent = `提交失败: ${res.status}`;
      return;
    }

    const data = await res.json();
    activeTaskId = data.task_id;
    taskIdEl.textContent = activeTaskId;
    submitMsg.textContent = `任务已创建: ${activeTaskId}`;

    if (pollTimer) clearInterval(pollTimer);
    pollTask();
    pollTimer = setInterval(pollTask, 2500);
  } catch (err) {
    submitMsg.textContent = `异常: ${err}`;
  }
});

alphaEl.addEventListener("input", () => {
  alphaValueEl.textContent = alphaEl.value;
  document.querySelectorAll("img.mask").forEach((img) => {
    img.style.opacity = alphaEl.value;
  });
});

downloadSegBtn.addEventListener("click", () => {
  if (!activeTaskId) return;
  const token = encodeURIComponent(tokenEl.value.trim());
  window.open(`/api/tasks/${activeTaskId}/seg?token=${token}`, "_blank");
});

downloadPdfBtn.addEventListener("click", () => {
  if (!activeTaskId) return;
  const token = encodeURIComponent(tokenEl.value.trim());
  window.open(`/api/tasks/${activeTaskId}/report?token=${token}`, "_blank");
});
