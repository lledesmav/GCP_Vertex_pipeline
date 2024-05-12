import os
import tempfile
from flask import Flask, render_template_string, render_template
import tensorflow_data_validation as tfdv
import tensorflow_metadata as tfmd
from tensorflow_data_validation.utils import display_util

import json

file                      = open('config.json')
config                    = json.load(file)

CREDENTIAL_PATH   = config["PROJECT_KEY_PATH"]
PROJECT_ID           = config["PIPELINE_PROJECT_ID"]
REGION            = config["PIPELINE_REGION"]
PATH         = config["PIPELINE_PATH_BUCKET"]+'/'+config["PIPELINE_METADATA"]["value"]
NAME_BUCKET       = config["PIPELINE_NAME_BUCKET"]


# Replace this with your DatasetFeatureStatisticsList object
from tensorflow_metadata.proto.v0 import statistics_pb2
from tensorflow_data_validation.utils import io_util
import tensorflow_data_validation as tfdv

# util function to load stats binary file from GCS
def load_stats_binary(input_path):
    stats_proto = statistics_pb2.DatasetFeatureStatisticsList()
    stats_proto.ParseFromString(io_util.read_file_to_string(input_path, binary_mode=True))
    return stats_proto

#Download file from Cloud Storage
def download_blob(bucket_name, object_name, destination_file_name):
    """Downloads a blob from the bucket."""
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.download_to_filename(destination_file_name)

from google.cloud import storage

from google.oauth2.service_account import Credentials
credentials = Credentials.from_service_account_file(CREDENTIAL_PATH)
client = storage.Client(credentials=credentials)

train_object = PATH + "/stats_anomalies/model/training_stats.pb"
download_blob(NAME_BUCKET, train_object,"train_stats.pb")
current_object = PATH + "/stats_anomalies/model/current_stats.pb"
download_blob(NAME_BUCKET, current_object,"current_stats.pb")

def all_models():
    training_model_stats = display_util.get_statistics_html(load_stats_binary('train_stats.pb'))
    current_model_stats = display_util.get_statistics_html(load_stats_binary('current_stats.pb'))

    titles = {
        'training_model': 'Training Model - Stats',
        'current_model': 'Current Model - Stats',
    }

    return render_template('all_models.html',
                           training_model_stats=training_model_stats,
                           current_model_stats=current_model_stats,
                           titles=titles)
