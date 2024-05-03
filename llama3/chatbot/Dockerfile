FROM python:3.9-slim

# Install software-properties-common to add new repository
RUN apt-get update && \
    apt-get install -y software-properties-common

# Upgrade pip to the latest version
# RUN python3.9 -m pip install --upgrade pip

# Install specific version of transformers and other packages
RUN pip install streamlit

# Make port 5000 available to the world outside this container
EXPOSE 5000

COPY chatbot.py /app/chatbot.py

# Run the inference script
CMD ["streamlit", "run", "/app/chatbot.py", "--server.port", "5000"]
