FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py .

EXPOSE 8501

# Shell form (not exec form) so $PORT is expanded - Render injects PORT at
# runtime and the container must bind to it, not a hardcoded 8501.
CMD streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}
