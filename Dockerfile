FROM python:3.7

# Skip the configuration part
ENV DEBIAN_FRONTEND noninteractive

# Update and install depedencies
RUN apt-get update \
    && apt-get install tesseract-ocr -y \
    && apt-get install -y wget unzip bc vim python3-pip libleptonica-dev git

RUN apt-get install ffmpeg libsm6 libxext6  -y

# Packages to complie Tesseract
RUN apt-get install -y --reinstall make && \
    apt-get install -y g++ autoconf automake libtool pkg-config \
    libpng-dev libjpeg62-turbo-dev libtiff5-dev libicu-dev \
    libpango1.0-dev autoconf-archive

RUN apt-get install -y poppler-utils

RUN mkdir /app

WORKDIR /app

RUN mkdir src && cd /app/src && \
    wget https://github.com/tesseract-ocr/tesseract/archive/refs/tags/5.1.0.zip && \
    unzip 5.1.0.zip && \
    cd /app/src/tesseract-5.1.0 && ./autogen.sh && ./configure && make && make install && ldconfig && \
    make training && make training-install && \
    cd /usr/local/share/tessdata && wget https://github.com/tesseract-ocr/tessdata_best/blob/main/eng.traineddata

RUN apt-get -q -y install tesseract-ocr-eng

RUN cp -R /usr/share/tesseract-ocr/4.00/tessdata/* /usr/local/share/tessdata/
# Setting the data prefix
ENV TESSDATA_PREFIX=/usr/local/share/tessdata/

RUN apt-get install -y poppler-utils

RUN tesseract --version

COPY . /app/

RUN pip install -r requirements.txt

#EXPOSE 5000

ENTRYPOINT ["python"]

CMD ["main.py"]
