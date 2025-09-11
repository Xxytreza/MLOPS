FROM python:3.11-slim

WORKDIR /app

COPY web_server/ ./web_server/


COPY regression.joblib ./

COPY houses.csv ./

RUN apt install sshpass

RUN pip install --no-cache-dir \
    fastapi[standard] \
    uvicorn[standard] \
    pydantic \
    joblib \
    pandas \
    scikit-learn

EXPOSE 6439

CMD ["fastapi","run", "web_server/server.py", "--port","6439"]
