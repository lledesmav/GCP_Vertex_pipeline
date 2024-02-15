FROM python:3.9
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
EXPOSE 8080
ENTRYPOINT ["python", "-m", "google.cloud.aiplatform.prediction.model_server"]
# The directory is created by root. This sets permissions so that any user can
# access the folder.
RUN mkdir -m 777 -p /usr/app /home
WORKDIR /usr/app
ENV HOME=/home
#RUN pip install --no-cache-dir --force-reinstall 'google-cloud-aiplatform[prediction]>=1.16.0'
ENV HANDLER_MODULE=google.cloud.aiplatform.prediction.handler
ENV HANDLER_CLASS=PredictionHandler
ENV PREDICTOR_MODULE=predictor
ENV PREDICTOR_CLASS=CprPredictor
# Copy the source directory into the docker container.
COPY [".", "."]
RUN pip install --no-cache-dir --force-reinstall -r requirements.txt