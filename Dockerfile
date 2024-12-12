FROM python:3.10

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

WORKDIR /src

ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8080"]
