FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get install -y ffmpeg libsm6 libxext6
COPY app /app
COPY requirements.txt /app
WORKDIR /app
RUN python3 -m pip install --break-system-packages -r requirements.txt
CMD ["python3", "app.py"]
EXPOSE 5000
