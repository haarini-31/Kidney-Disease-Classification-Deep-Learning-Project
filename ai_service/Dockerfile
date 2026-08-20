FROM python:3.10-slim

WORKDIR /app

# Install compilation tools for python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY ai_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI application files
COPY ai_service/app.py .
COPY ai_service/prediction.py .

# Copy model parts from project artifacts
COPY artifacts/training/model.h5.part-0 .
COPY artifacts/training/model.h5.part-1 .

# Reconstruct model during image build using python binary file IO to avoid shell-dependent command execution
RUN python -c "\
import glob;\
part_files = sorted(glob.glob('model.h5.part-*'));\
print(f'Reconstructing model from: {part_files}');\
with open('model.h5', 'wb') as outfile:\
    for part in part_files:\
        with open(part, 'rb') as infile:\
            outfile.write(infile.read());\
" && rm model.h5.part-*

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
