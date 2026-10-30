# Web 演示系统

给一条酒店评论，系统实时判断它是「好评」还是「差评」并给出置信度。

- 前端：Vue 3（JavaScript）+ Vite，页面源码在 `frontend/`
- 后端：FastAPI（Python），单文件 `main.py`
- 部署形态：单服务——FastAPI 一个进程同时提供预测接口和前端页面，**一条命令启动**

## 功能

| 功能 | 说明 |
|------|------|
| 评论分析 | 输入评论点「分析」，返回好评/差评结论与置信度（好评绿色、差评红色） |
| 批量示例 | 内置 7 条典型例句（含「但是 / 可惜 / 却」转折句式），一键填入 |
| 历史记录 | 浏览器本地保存最近 50 条分析记录，刷新页面不丢失 |
| 模型成绩单 | 展示 LSTM 与 BERT 的测试集准确率对比和混淆矩阵图 |

## 运行（答辩演示照此步骤）

页面文件已提前构建好放在 `static/`，**演示机不需要安装 Node**，只要 Python 环境（复用模型侧的虚拟环境）：

```bash
cd model
source .venv/bin/activate            # 复用模型侧虚拟环境
uv pip install -r ../web/requirements.txt

cd ../web
uvicorn main:app --port 8000         # 首次启动需加载模型，约 10 秒
```

浏览器打开 `http://127.0.0.1:8000` 即可使用。

> 注意：必须保持整个项目目录结构完整——后端按相对位置加载 `model/saved/bert/` 的模型文件和 `model/reports/` 的评估图表，两者缺一页面会提示对应内容不可用。

## 开发（只有改前端时才需要）

```bash
cd web/frontend
npm install
npm run dev        # 开发模式，页面热更新，接口自动转发给本机 8000 端口的后端
npm run build      # 改完后重新构建，产物自动写入 ../static/
```

## 与模型侧的衔接

- 预测：服务启动时一次性加载 `model/saved/bert/` 的模型进内存，只做推理，不含任何训练逻辑
- 成绩单：直接读取 `model/reports/` 下的指标文件与混淆矩阵图，与评估报告同源，数字不会出现两套
