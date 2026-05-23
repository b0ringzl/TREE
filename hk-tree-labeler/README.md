# 香港城市树木街景数据按需采集与标注系统

本项目所有输入、输出、缓存和运行文件均约束在 `D:\TREE` 下：

- 树种与坐标输入：`D:\TREE\选取树种\`
- 待审核街景缓存：`D:\TREE\temp\{Tree_ID}\`
- YOLO 数据集输出：`D:\TREE\dataset\{Species}\{Tree_ID}\`

## 后端启动

推荐使用一键会话启动器。它会询问 API Key，自动启动后端、前端和专用浏览器窗口；关闭该窗口后，会自动结束服务并输出本次工作报告。

```powershell
cd D:\TREE\hk-tree-labeler
powershell -ExecutionPolicy Bypass -File .\run-session.ps1
```

也可以直接双击或运行：

```powershell
D:\TREE\hk-tree-labeler\start-labeler.bat
```

也可以用参数直接传入 API Key：

```powershell
powershell -ExecutionPolicy Bypass -File .\run-session.ps1 -ApiKey "你的 Google Maps API Key"
```

以下是手动分开启动方式。

```powershell
cd D:\TREE\hk-tree-labeler\backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:GOOGLE_MAPS_API_KEY="你的 Google Maps API Key"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## 前端启动

```powershell
cd D:\TREE\hk-tree-labeler\frontend
npm install
npm run dev
```

打开 Vite 输出的本地地址，通常是 `http://127.0.0.1:5173`。

## API

- `GET /api/species/list`：扫描 `D:\TREE\选取树种` 并返回树种目录名。
- `POST /api/task/start`：传入 `{ "species": "Ficus microcarpa", "target_count": 50 }` 后开始按需缓存。
- `GET /api/task/next`：取得下一株待标注树木的 3 张街景图。
- `POST /api/task/submit`：保存 3 张图的 YOLO 归一化框，移动到 `dataset`。
- `POST /api/task/reject`：删除 `temp` 下当前样本。

## 标注交互

- 拖动框体：移动。
- 拖动四角：缩放。
- 在空白处拖动：新建框。
- `Delete`：删除当前选中框。
- `Space`：保存并加载下一株。
- `R`：拒绝当前样本并加载下一株。
