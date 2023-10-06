FROM python:3.10.5
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD streamlit run airline_app.py