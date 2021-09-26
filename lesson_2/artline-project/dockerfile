
RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY app/app.ini .

RUN wget https://www.dropbox.com/s/04suaimdpru76h3/ArtLine_920.pkl?dl=1 
RUN mv ArtLine_920.pkl?dl=1 ArtLine_920.pkl

COPY . .

RUN pip install -r requirements.txt

CMD ["uwsgi", "app.ini"]