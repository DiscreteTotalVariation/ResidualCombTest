FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils fonts-urw-base35 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
ENV MPLBACKEND=Agg PYTHONWARNINGS=ignore
COPY code/ ./code/
COPY figures/ ./figures/
COPY tables/ ./tables/
COPY pipeline.sh .
RUN mkdir -p results paper/img data
ENTRYPOINT ["bash", "pipeline.sh"]
