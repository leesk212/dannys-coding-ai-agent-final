FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY .env.example .env.example

# Install Python deps
RUN pip install --no-cache-dir .

# Memory persistence volume
RUN mkdir -p /root/.coding_agent/memory
VOLUME /root/.coding_agent/memory

# DeepAgents memory
RUN mkdir -p /root/.deepagents/coding-agent
VOLUME /root/.deepagents

# Streamlit port
EXPOSE 8501

# Default: WebUI mode
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["python", "-m", "coding_agent", "--webui"]
