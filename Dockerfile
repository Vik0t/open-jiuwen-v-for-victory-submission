FROM mirror.gcr.io/library/python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY submission/requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /app/requirements.lock
COPY math_requirements.lock /app/math_requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /app/math_requirements.lock
COPY scripts/patch_sympy.py /tmp/patch_sympy.py
RUN python /tmp/patch_sympy.py && rm /tmp/patch_sympy.py
COPY submission /app/submission
COPY run_agent.py /app/run_agent.py
ENTRYPOINT ["python", "/app/run_agent.py"]
