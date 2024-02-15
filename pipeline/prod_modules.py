from CustomLib.gcp import cloudstorage, cloudbuild, bigquery
from typing import Dict, Optional, Sequence, NamedTuple, List, Union, Tuple
import pandas as pd

def build_custom_predict_routine_image(BASE_IMAGE             : str, 
                                       CUST_PRED_ROUTINE_PATH : str,
                                      NAME_BUCKET             : str) -> Tuple[str, str]:
    #==== Build the Custom Predict Routine ====#
    out, err = cloudbuild.submit(image_name = BASE_IMAGE, 
                                 code_path  = CUST_PRED_ROUTINE_PATH,
                                name_bucket = NAME_BUCKET)
    return out, err

def get_endpoint_by_display_name(display_name: str):
    # List all available endpoints
    endpoints = aiplatform.Endpoint.list()

    # Filter the endpoints by the given display_name
    for endpoint in endpoints:
        if endpoint.display_name == display_name:
            return endpoint
    
    return None


from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
def predict_custom_trained_model_sample(project: str,
                                        endpoint_id: str,
                                        instances: Union[Dict, List[Dict]],
                                        location: str = "us-central1",
                                        api_endpoint: str = "us-central1-aiplatform.googleapis.com",):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(project=project, 
                                    location=location, 
                                    endpoint=endpoint_id)
    response = client.predict(endpoint=endpoint, 
                              instances=instances, 
                              parameters=parameters)

    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    
    return predictions

def get_model_by_display_name(display_name, project, location):
    client = aiplatform.gapic.ModelServiceClient(client_options={"api_endpoint": f'{location}-aiplatform.googleapis.com'})
    model_list = client.list_models(parent=f"projects/{project}/locations/{location}")

    model = None
    for model_item in model_list:
        if model_item.display_name == display_name:
            model = aiplatform.Model(model_name=model_item.name)
            break
            
    return model

def last_slash_detec(path : str):
    # Find the position of the last '/' in the string
    last_slash_index = path.rfind('/')

    # If a '/' is found, remove everything after the last '/'
    if last_slash_index != -1:
        path_final = path[:last_slash_index]
    else:
        path_final = path
    return path_final

def batch_prediction_request(project           = str,
                             location          = str,
                             model_name        = str,
                             input_path        = str,
                             output_path       = str,
                             labels            = Dict,
                             prod_config       = Dict,
                             training_dataset  = str,
                             features          = list,
                             target            = str,
                             name_bucket       = str,
                             path_bucket       = str,
                             df_to_predict     = pd.DataFrame,
                             credential_path   = str):
    
    DEPLOY_COMPUTE                      = prod_config["BATCH_PREDICTION"]["DEPLOY_COMPUTE"]
    START_REPLICA                       = prod_config["BATCH_PREDICTION"]["START_REPLICA"]
    MAX_REPLICA                         = prod_config["BATCH_PREDICTION"]["MAX_REPLICA"]
    SAMPLE_RATE                         = prod_config["BATCH_PREDICTION"]["SAMPLE_RATE"]
    skew_thresholds                     = prod_config["BATCH_PREDICTION"]['SKEW_THRESHOLDS']
    attribution_score_skew_thresholds   = prod_config["BATCH_PREDICTION"]['ATTRIBUTION_SCORE_SKEW_THRESHOLDS']
    drift_thresholds                    = prod_config["BATCH_PREDICTION"]['DRIFT_THRESHOLDS']
    attribution_score_drift_thresholds  = prod_config["BATCH_PREDICTION"]['ATTRIBUTION_SCORE_DRIFT_THRESHOLDS']
    ALERT_MAILS                         = prod_config["BATCH_PREDICTION"]["ALERT_MAILS"]
    
    df_to_predict.to_json(input_path+'/input_batch_model'+'.jsonl', orient='records', lines=True)   #REVISAR SI ES NECESARIO PERMISOS
    INPUT_URI = (input_path + '/input_batch_model' + '.jsonl').replace('/gcs/', 'gs://')
    OUTPUT_URI = (output_path+'/predictions').replace('/gcs/', 'gs://')
    training_dataset = training_dataset.replace('/gcs/', 'gs://')
    
    from google.cloud.aiplatform_v1beta1 import (BatchPredictionJob, GcsSource, GcsDestination, MachineSpec, ModelMonitoringConfig, ModelMonitoringObjectiveConfig, SamplingStrategy, ThresholdConfig, ModelMonitoringAlertConfig, BatchDedicatedResources)
    from google.cloud import aiplatform
    
    MODEL             = aiplatform.Model(model_name = model_name)
    DISPLAY_NAME      = 'batch-' + MODEL.display_name
    INPUT_FORMAT      = 'jsonl'
    PREDICTION_FORMAT = 'jsonl'
    TRAINING_FORMAT   = 'csv'
    STRATEGY          = SamplingStrategy.RandomSampleConfig(sample_rate=SAMPLE_RATE)
    
    # Create the variables
    SKEW_THRESHOLDS                    = {feature: ThresholdConfig(value=skew_thresholds[feature]) \
                                          for feature in features}
    ATTRIBUTION_SCORE_SKEW_THRESHOLDS  = {feature: ThresholdConfig(value=attribution_score_skew_thresholds[feature]) \
                                          for feature in features}
    DRIFT_THRESHOLDS                   = {feature: ThresholdConfig(value=drift_thresholds[feature]) \
                                          for feature in features}
    ATTRIBUTION_SCORE_DRIFT_THRESHOLDS = {feature: ThresholdConfig(value=attribution_score_drift_thresholds[feature]) \
                                          for feature in features}
    STATS_ANOMALIES_URI                = 'gs://'+name_bucket+'/'+path_bucket+'/'+'stats_anomalies'+'/model'
    ALERT_CONFIG                       = ModelMonitoringAlertConfig(email_alert_config = ModelMonitoringAlertConfig.EmailAlertConfig(user_emails = ALERT_MAILS), 
                                                                    enable_logging = True)
    
    MODEL_MONITORING_CONFIG = ModelMonitoringConfig(
                            objective_configs = [ModelMonitoringObjectiveConfig(
                                            training_dataset = ModelMonitoringObjectiveConfig.TrainingDataset(
                                                                gcs_source                = GcsSource(uris = [training_dataset]),
                                                                data_format               = TRAINING_FORMAT,
                                                                target_field              = target,
                                                                logging_sampling_strategy = SamplingStrategy(random_sample_config = STRATEGY)),
                                            training_prediction_skew_detection_config = ModelMonitoringObjectiveConfig.TrainingPredictionSkewDetectionConfig(
                                                                skew_thresholds                   = SKEW_THRESHOLDS,
                                                                attribution_score_skew_thresholds = ATTRIBUTION_SCORE_SKEW_THRESHOLDS),
                                            prediction_drift_detection_config = ModelMonitoringObjectiveConfig.PredictionDriftDetectionConfig(
                                                                drift_thresholds                   = DRIFT_THRESHOLDS,
                                                                attribution_score_drift_thresholds = ATTRIBUTION_SCORE_DRIFT_THRESHOLDS))],
                            alert_config = ALERT_CONFIG,
                            stats_anomalies_base_directory = GcsDestination(output_uri_prefix = STATS_ANOMALIES_URI))
    
    BATCH_PREDICTION_JOB = BatchPredictionJob(
                        display_name            = DISPLAY_NAME,
                        model                   = MODEL.versioned_resource_name,
                        input_config            = BatchPredictionJob.InputConfig(
                                                    instances_format=INPUT_FORMAT,
                                                    gcs_source=GcsSource(uris=[INPUT_URI])),
                        output_config           = BatchPredictionJob.OutputConfig(
                                                    predictions_format=PREDICTION_FORMAT,
                                                    gcs_destination=GcsDestination(output_uri_prefix=OUTPUT_URI)),
                        dedicated_resources     = BatchDedicatedResources(
                                                    machine_spec=MachineSpec(machine_type=DEPLOY_COMPUTE),
                                                    starting_replica_count=START_REPLICA,
                                                    max_replica_count=MAX_REPLICA),
                        labels                  = labels,
                        model_monitoring_config = MODEL_MONITORING_CONFIG)
    
    from google.cloud.aiplatform_v1beta1.services.job_service import JobServiceClient
    API_ENDPOINT = f"{location}-aiplatform.googleapis.com"
    from google.oauth2.service_account import Credentials
    credentials = Credentials.from_service_account_file(credential_path)
    client = JobServiceClient(client_options = {"api_endpoint": API_ENDPOINT},
                              credentials = credentials)
    job = client.create_batch_prediction_job(parent               = f"projects/{project}/locations/{location}",
                                             batch_prediction_job = BATCH_PREDICTION_JOB,
                                             timeout              = 3600)
    
    from google.cloud.aiplatform_v1beta1 import GetBatchPredictionJobRequest
    request  = GetBatchPredictionJobRequest(name = job.name)
    response = client.get_batch_prediction_job(request = request, 
                                               timeout = 3600)
    
    print('EL JOB.NAME ES : '+job.name.split('/')[-1])
    import time
    while response.state.real != 4:
        time.sleep(15)
        response = client.get_batch_prediction_job(request = request, 
                                                   timeout = 3600)
        if response.state.real == 4:
            pass
    
    #===== Review all the results predictions "prediction.results-0000x-of-0000y" =====#
    import pandas as pd
    import os
    from CustomLib.gcp import cloudstorage
    
    # List all the JSON files in the directory
    prefix      = response.output_info.gcs_output_directory+'/'
    blobs=cloudstorage.directory_list(prefix)
    blobs=(str(blobs[0]).replace("b'","").replace("'","")).split("\\n")

    dataframes = []
    for blob in blobs:
        if "prediction.results" in blob:
            # Download the JSON file as a string
            json_content = cloudstorage.read_txt_as_str(blob)
            # Read the JSON string into a DataFrame
            df = pd.read_json(json_content, lines = True)
            # Append the DataFrame to the list
            dataframes.append(df)
    
    # Combine all the DataFrames into a single DataFrame
    result = pd.concat(dataframes, ignore_index = True)
    
    df_to_predict[target] = list(result['prediction'])
    job_name = str(job.name.split('/')[-1])
    
    return df_to_predict, job_name


def move_stats_file(job_name   : str, 
                    name_bucket : str, 
                    path_bucket : str):
    import subprocess
    import os
    #==== Wait until the statistical analysis files are created from the last model ====#
    import time
    path_1 = f'gs://{name_bucket}/{path_bucket}/stats_anomalies/model/job-{job_name}/bp_monitoring/stats_training/stats/training_stats'
    path_2 = f'gs://{name_bucket}/{path_bucket}/stats_anomalies/model/job-{job_name}/bp_monitoring/stats_and_anomalies/stats/current_stats'
    
    while True:
        path_1_exists = cloudstorage.check_file_exists(path_1)
        path_2_exists = cloudstorage.check_file_exists(path_2)
        if (path_1_exists and path_2_exists):
            break
        time.sleep(30)

    #===============================================================#
    
    input_train_bucket    = f'gs://{name_bucket}/{path_bucket}/stats_anomalies/model/job-{job_name}/bp_monitoring/stats_training/stats/training_stats'
    input_current_bucket  = f'gs://{name_bucket}/{path_bucket}/stats_anomalies/model/job-{job_name}/bp_monitoring/stats_and_anomalies/stats/current_stats'
    
    output_train_bucket   = f'gs://{name_bucket}/{path_bucket}/stats_anomalies/model/training_stats.pb'
    output_current_bucket = f'gs://{name_bucket}/{path_bucket}/stats_anomalies/model/current_stats.pb'
    
    _ = cloudstorage.copy_file(source      = input_train_bucket.strip(), 
                               destination = output_train_bucket.strip())
    _ = cloudstorage.copy_file(source      = input_current_bucket.strip(), 
                               destination = output_current_bucket.strip())
    
    
def build_monitoring_prediction_image(labels      : Dict, 
                                      project     : str, 
                                      location    : str, 
                                      name_bucket : str, 
                                      path_bucket : str):
    import subprocess
    # Create Docker image and send it to Artifact Registry
    repo_name        = list(labels.values())[0]
    container        = list(labels.values())[0]+'-monitor-pred'
    MONITORING_IMAGE = '{}-docker.pkg.dev/{}/{}/{}:latest'.format(location, project, repo_name ,container)
    
    # Update the JSON file for monitoring_prediction
    import json
    update_json={"PROJECT_ID"  : project,
                 "REGION"      : location,
                 "NAME_BUCKET" : name_bucket,
                 "PATH_BUCKET" : path_bucket}
    with open("pipeline/monitoring_prediction/config.json", "w") as outfile:
        json.dump(update_json, outfile)
        
    # Create the build
    out, err = cloudbuild.submit(image_name = MONITORING_IMAGE,
                                   code_path = "pipeline/monitoring_prediction",
                                   name_bucket = name_bucket,
                                   project_id = project)
    
    return MONITORING_IMAGE
        