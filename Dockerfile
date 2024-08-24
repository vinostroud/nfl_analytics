# Use an official Python runtime as a parent image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app/
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock


# Install Poetry
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi


# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port that the app runs on
EXPOSE 8501

# Define environment variable
ENV NAME nfl_analytics

# Run the application
CMD ["streamlit", "run", "src/app_fe.py"]
