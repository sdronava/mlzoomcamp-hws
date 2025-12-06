#FROM agrigorev/model-2024-hairstyle:v3
FROM agrigorev/model-2025-hairstyle:v1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
COPY pyproject.toml uv.lock ./
RUN uv pip install --system -r <(uv export --format requirements-txt)
RUN pip install onnxruntime keras-image-helper==0.0.2

ARG MODEL_NAME=hair_classifier_empty.onnx
ENV MODEL_NAME=${MODEL_NAME}

COPY lambda_function.py ./

CMD ["lambda_function.lambda_handler"]

