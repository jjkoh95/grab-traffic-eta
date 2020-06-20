# Use the official lightweight Python image.
# https://hub.docker.com/_/python
# slim is way way better for pip wheel
FROM python:3.7-slim

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# gcc, musl-dev required by models
RUN apt-get update && \
    apt-get -y install gcc musl-dev

# Install production dependencies.
RUN pip install --upgrade pip setuptools wheel

RUN pip install Flask gunicorn

# pytorch dependencies
# since we only need CPU version here
RUN pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install -r requirements.txt


EXPOSE 80

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD exec gunicorn --bind :80 --workers 2 --threads 8 --timeout 0 app:app