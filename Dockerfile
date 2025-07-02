# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# First copy only the files needed for downloading models
COPY app/__init__.py /app/app/__init__.py
COPY app/services/__init__.py /app/app/services/__init__.py
COPY app/services/ocr.py /app/app/services/ocr.py
COPY download_models.py /app/download_models.py

# Run the download script to populate the cache
# This layer will be cached as long as the download-related files don't change.
RUN python download_models.py

# Now copy the rest of the application
COPY . /app

# Run uvicorn when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
