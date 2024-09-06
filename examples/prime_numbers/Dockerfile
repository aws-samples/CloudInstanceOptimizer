FROM python:3.11

RUN apt-get update
RUN apt-get install software-properties-common -y

COPY requirements.txt /
RUN python -m pip install -r requirements.txt

COPY ./src/utils.py utils.py
COPY ./src/monitor_system.py monitor_system.py

# Code a user wants to run on each EC2 instance and find the best option
COPY ./examples/prime_numbers/stress_test.py stress_test.py

WORKDIR /
