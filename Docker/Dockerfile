#FROM tiangolo/uwsgi-nginx-flask:python3.7
FROM tiangolo/uwsgi-nginx-flask:python3.6
#FROM python:3.6.2
#COPY ./app /app

ADD . /code
WORKDIR /code
ADD modelDogCat.h5 /code
ADD requirements.txt /code
ADD main.py /code
ADD /templates /code
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#EXPOSE 5000
#CMD ['python', 'main.py']