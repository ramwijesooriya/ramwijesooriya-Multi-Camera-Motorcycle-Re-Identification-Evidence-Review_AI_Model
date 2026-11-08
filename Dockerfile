FROM python:3.11-slim AS backend
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data YOLO_WEIGHTS=/models/yolo11n.pt YOLO_CONFIG_DIR=/tmp/ultralytics
ENV TORCH_HOME=/models/torch
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt backend/constraints-docker-cpu.txt ./
# Official CPython 3.11 CPU wheels exist for Linux ARM64 and AMD64.
# Fail on unsupported targets rather than falling back to a CUDA build or source.
RUN python -c "import platform, sys; assert sys.platform == 'linux' and platform.machine() in ('aarch64', 'x86_64'), 'CPU wheel installation supports Linux ARM64/AMD64 only'" \
    && python -m pip install --no-cache-dir --only-binary=:all: \
        --index-url https://download.pytorch.org/whl/cpu \
        --constraint constraints-docker-cpu.txt torch torchvision \
    && python -m pip install --no-cache-dir \
        --constraint constraints-docker-cpu.txt -r requirements.txt \
    && python -m pip check \
    && python -c "import importlib.metadata as m; import torch, torchvision; from torchvision.ops import nms; assert torch.version.cuda is None, 'Expected CPU-only torch'; unwanted = [d.metadata['Name'] for d in m.distributions() if d.metadata['Name'].lower().replace('_', '-').startswith(('nvidia-', 'cuda-', 'triton'))]; assert not unwanted, unwanted; assert nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), 0.5).tolist() == [0]; print('Validated CPU torch', torch.__version__, 'torchvision', torchvision.__version__)"
# Cache real pretrained weights during build. A failed download fails the build.
RUN mkdir -p /models && cd /models && python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
RUN python -c "from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights; mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)"
COPY backend/app ./app
RUN useradd --create-home appuser && mkdir -p /data && chown appuser:appuser /data
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM node:22-alpine AS ui-build
WORKDIR /ui
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine AS frontend
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=ui-build /ui/dist /usr/share/nginx/html
EXPOSE 80
