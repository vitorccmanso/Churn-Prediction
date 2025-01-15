FROM python:3.12
WORKDIR /client_churn_pred
COPY /app /client_churn_pred/app
RUN pip install -r /client_churn_pred/app/requirements_app.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]