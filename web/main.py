"""Web 演示系统后端：加载 model/saved/bert 的模型做情感预测，并托管前端页面。

接口：
    POST /api/predict    输入一条评论文本，返回「好评/差评 + 置信度」
    GET  /api/metrics    返回 LSTM 与 BERT 的测试集指标（读自 model/reports/）
    GET  /api/matrix/{lstm|bert}   返回对应的混淆矩阵图
    其余路径             托管 static/ 下的前端构建产物（Vue 打包结果）

运行（复用模型侧虚拟环境，先安装本目录依赖）：
    uv pip install -r requirements.txt
    uvicorn main:app --port 8000
    浏览器打开 http://127.0.0.1:8000
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import BertTokenizerFast, TFBertForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "model" / "saved" / "bert"
REPORTS_DIR = BASE_DIR.parent / "model" / "reports"
STATIC_DIR = BASE_DIR / "static"

MAX_LENGTH = 128  # 与训练时保持一致

tf.get_logger().setLevel("ERROR")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时把模型加载进内存，之后所有请求共用，不再重复加载。"""
    print(f"[模型] 正在加载 {MODEL_DIR} ...")
    app.state.tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    app.state.model = TFBertForSequenceClassification.from_pretrained(MODEL_DIR)
    print("[模型] 加载完成，可以开始预测")
    yield


app = FastAPI(title="酒店评论情感分析系统", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str


@app.post("/api/predict")
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="评论文本不能为空")

    enc = app.state.tokenizer(
        [text], truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="tf"
    )
    logits = app.state.model.predict(dict(enc), verbose=0).logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    positive_prob = float(probs[1])
    return {
        "label": "好评" if positive_prob >= 0.5 else "差评",
        "confidence": round(max(positive_prob, 1 - positive_prob), 4),
        "positive_prob": round(positive_prob, 4),
    }


@app.get("/api/metrics")
def metrics():
    """看板数据：直接读模型侧的评估产物，不手工复制数字。"""
    result = {}
    for name in ("lstm", "bert"):
        path = REPORTS_DIR / f"{name}_metrics.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            result[name] = {
                "model": data.get("model", name.upper()),
                "test_accuracy": data.get("test_accuracy"),
                "best_epoch": data.get("best_epoch"),
                "history": data.get("history", {}),
            }
    if not result:
        raise HTTPException(status_code=404, detail="未找到模型指标文件，请先完成模型训练")
    return result


@app.get("/api/matrix/{name}")
def confusion_matrix(name: str):
    """混淆矩阵图，只允许 lstm / bert 两个固定文件名。"""
    if name not in ("lstm", "bert"):
        raise HTTPException(status_code=404, detail="无此图表")
    path = REPORTS_DIR / f"confusion_matrix_{name}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="图表文件不存在，请先运行 evaluate.py")
    return FileResponse(path, media_type="image/png")


STATIC_DIR.mkdir(exist_ok=True)  # 前端未构建时保证服务仍能启动
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
