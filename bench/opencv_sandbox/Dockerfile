FROM python:3.12-slim
RUN python -m pip install --no-cache-dir numpy==2.2.6 opencv-python-headless==4.12.0.88
COPY measure.py /app/measure.py
WORKDIR /work
USER 65534:65534
ENTRYPOINT ["python", "-I", "/app/measure.py"]
