from kfp.dsl import pipeline, component, Dataset, InputPath, OutputPath, Model, Input, Output, Metrics, Artifact, Condition, ClassificationMetrics
from typing import Dict, Optional, Sequence, NamedTuple, List, Union
from kfp import components
from kfp import compiler
from google.cloud import storage
from google.cloud import aiplatform
from datetime import datetime

########################################################################################
#=========================  Get VARIABLES & BASE_IMAGE   ==============================#
########################################################################################
import json
import os

file                      = open('config.json')
config                    = json.load(file)

CREDENTIAL_PATH   = config["PROJECT_KEY_PATH"]
PROJECT           = config["PIPELINE_PROJECT_ID"]
REGION            = config["PIPELINE_REGION"]
LABELS            = {config["PIPELINE_METADATA"]["key"]:config["PIPELINE_METADATA"]["value"]}
MODEL_NAME        = config["PIPELINE_MODEL_NAMES"]
MODEL_DESCRIPTION = config["PIPELINE_MODEL_DESCRIPTION"]
PATH_BUCKET         = config["PIPELINE_PATH_BUCKET"]+'/'+config["PIPELINE_METADATA"]["value"]
NAME_BUCKET       = config["PIPELINE_NAME_BUCKET"]
COMPILE_NAME_FILE = "pred-"+config["PIPELINE_METADATA"]["value"]+'.yaml'
TIMESTAMP         = datetime.now().strftime("%Y%m%d%H%M%S")
DISPLAY_NAME      = config["PIPELINE_METADATA"]["key"]+"-"+config["PIPELINE_METADATA"]["value"]+'-pred-{}'.format(TIMESTAMP)

OUTPUT_BQ_PROJECT = config["OUTPUT_PROJECT_ID"]
OUTPUT_BQ_DATASET = config["OUTPUT_DATASET_ID"]
OUTPUT_BQ_TABLE   = config["OUTPUT_TABLE_ID"]

file        = open('pipeline/prod_config.json')
PROD_CONFIG = json.load(file)

BASE_CONTAINER_IMAGE_NAME = config["PIPELINE_METADATA"]["value"]
BASE_IMAGE                = '{}-docker.pkg.dev/{}/{}/{}:'.format(REGION, 
                                                                 PROJECT, 
                                                                 BASE_CONTAINER_IMAGE_NAME, 
                                                                 BASE_CONTAINER_IMAGE_NAME)+'latest'

from google.oauth2.service_account import Credentials
credentials = Credentials.from_service_account_file(CREDENTIAL_PATH)
aiplatform.init(project=PROJECT, location=REGION, credentials=credentials)

##############################################################################################
#===================================== get_data COMPONENT ===================================#
##############################################################################################
@component(base_image = BASE_IMAGE)
def get_data(credential_path : str,
             name_bucket : str,
             path_bucket : str,
             dataset         : OutputPath("Dataset")) -> NamedTuple("output", [("features", list),("target", str),("train_data_path",str)]):
    
    #==== Define credentials ====#
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    
    #==== Importing necessary libraries ====#
    import pandas as pd
    from sklearn.datasets import load_iris
    
    #==== Loading the Iris dataset ====#
    iris = load_iris()
    data = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
    data['target'] = iris['target']
    
    # Define a mapping dictionary for the columns
    column_mapping = {
        'sepal length (cm)': 'sepal_length_cm',
        'sepal width (cm)': 'sepal_width_cm',
        'petal length (cm)': 'petal_length_cm',
        'petal width (cm)': 'petal_width_cm',
    }

    # Rename the columns using the mapping
    data.rename(columns=column_mapping, inplace=True)
    
    columns = list(data.columns)
    features = columns[:-1]
    target = columns[-1]

    #==== Getting a random sample of 100 rows ====#
    # If the dataset contains fewer than 100 rows, this will return all rows in random order
    random_sample = data.sample(n=min(100, len(data)), random_state=42) # The random_state ensures reproducibility
    data = random_sample.drop(columns=['target'])
    
    #==== Get the training_dataset ====#
    from CustomLib.gcp import cloudstorage
    gcs_path = f"gs://{name_bucket}/{path_bucket}/last_training_path.txt"
    train_data_path = cloudstorage.read_txt_as_str(gcs_path=gcs_path)
    
    #==== Save the df in GCS ====#
    cloudstorage.write_csv(df       = data, 
                           gcs_path = dataset + '.csv')
    
    return features, target, train_data_path
    
    
#########################################################################################
#======================== predict_model_endpoint COMPONENT =============================#
#########################################################################################
@component(base_image = BASE_IMAGE)
def predict_model_endpoint(credential_path : str,
                           data_input      : InputPath("Dataset"),
                           model_name      : str) -> List:
    
    #==== Define credentials ====#
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    
    from google.oauth2.service_account import Credentials
    credentials = Credentials.from_service_account_file(credential_path)
    aiplatform.init(project=PROJECT, location=REGION, credentials=credentials)
    
    #==== Read input data from GCS ====#
    from CustomLib.gcp import cloudstorage
    data = cloudstorage.read_csv_as_df(gcs_path = data_input + '.csv')
    
    # Converting the DataFrame to a list of lists
    instances = data.values.tolist()
    
    # Get the endpoint
    from pipeline.prod_modules import get_endpoint_by_display_name
    endpoint = get_endpoint_by_display_name(display_name = model_name + "-endpoint")
    
    response = endpoint.predict(instances=instances)
    predictions = response.predictions

    return predictions

#########################################################################################
#============================== save_predictions COMPONENT =============================#
#########################################################################################
@component(base_image = BASE_IMAGE)
def save_predictions(credential_path   : str,
                     data_input        : InputPath("Dataset"),
                     predictions       : List,
                     output_bq_project : str,
                     output_bq_dataset : str,
                     output_bq_table   : str):
    
    #==== Define credentials ====#
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    
    #==== Read input data from GCS ====#
    from CustomLib.gcp import cloudstorage
    data = cloudstorage.read_csv_as_df(gcs_path = data_input + '.csv')
    # Ad the predictions column
    data['predictions'] = predictions
    
    import json
    file         = open('config.json')
    config       = json.load(file)
    schema_dict  = config["OUTPUT_SCHEMA"]
    
    #==== Save predictions to bigquery ====#
    from CustomLib.gcp import bigquery
    bigquery.write_df(df              = data, 
                      project_id      = output_bq_project, 
                      dataset_id      = output_bq_dataset, 
                      table_id        = output_bq_table,
                      credential_path = credential_path,
                      schema_dict     = schema_dict, 
                      if_exists       = 'replace')
    
    
###################################################################################
#============================ batch_predict COMPONENT ============================#
###################################################################################
@component(base_image = BASE_IMAGE)
def batch_predict(credential_path  : str,
                  project          : str,
                  location         : str,
                  name_bucket      : str,
                  path_bucket      : str,
                  labels           : Dict,
                  prod_config      : Dict,
                  model_name       : str,
                  features         : list,
                  target           : str,
                  train_data_path  : str,
                  data_input       : InputPath("Dataset"), 
                  data_output      : OutputPath("Dataset")) -> NamedTuple("output", [("job_name", str)]):
    
    #==== Define credentials ====#
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    
    from google.oauth2.service_account import Credentials
    credentials = Credentials.from_service_account_file(credential_path)
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=location, credentials=credentials)
    
    #==== Read input data from GCS ====#
    from CustomLib.gcp import cloudstorage
    df = cloudstorage.read_csv_as_df(gcs_path = data_input + '.csv')
    
    #==== Get the model ====#
    from pipeline.prod_modules import get_model_by_display_name
    model = get_model_by_display_name(display_name = model_name, 
                                      project      = project, 
                                      location     = location)
    
    #==== Generate batch predictions ====#
    from pipeline.prod_modules import last_slash_detec, batch_prediction_request
    df_to_predict, job_name = batch_prediction_request(project           = project,
                                                       location          = location,
                                                       model_name        = model.name,
                                                       input_path        = last_slash_detec(data_input),
                                                       output_path       = last_slash_detec(data_output),
                                                       labels            = labels,
                                                       prod_config       = prod_config,
                                                       training_dataset  = train_data_path,
                                                       features          = features,
                                                       target            = target,
                                                       name_bucket       = name_bucket,
                                                       path_bucket       = path_bucket,
                                                       df_to_predict     = df,
                                                       credential_path   = credential_path)
    
    #==== Save predictions ====#
    cloudstorage.write_csv(df       = df_to_predict, 
                           gcs_path = data_output +'.csv')
    
    Output = NamedTuple("output", [("job_name", str)])  # Define the NamedTuple
    result = Output(job_name=job_name)  # Create an instance with the job_name
    
    return result

    
################################################################################
#============================ save_stats COMPONENT ============================#
################################################################################
@component(base_image = BASE_IMAGE)
def save_stats(credential_path  : str,
               project          : str,
               location         : str,
               name_bucket      : str,
               path_bucket      : str,
               labels           : Dict,
               job_name         : str):
    
    # Define credentials
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    os.environ['GOOGLE_CLOUD_PROJECT'] = project
    
    # Move stats files in GCS
    from pipeline.prod_modules import move_stats_file, build_monitoring_prediction_image
    move_stats_file(job_name    = job_name, 
                    name_bucket = name_bucket, 
                    path_bucket = path_bucket)
    
    # Create Cloud Run Service
    from CustomLib.gcp import cloudrun
    cloud_run_name = list(labels.values())[0]+'-monitor-pred'

    #==== Create docker image and save in Artifact Registry ====#
    MONITORING_IMAGE = build_monitoring_prediction_image(labels      = labels, 
                                                         project     = project, 
                                                         location    = location, 
                                                         name_bucket = name_bucket, 
                                                         path_bucket = path_bucket)
    # Create Cloud Run service
    out, err = cloudrun.service_create(service_name = cloud_run_name, 
                                location     = location, 
                                image_name   = MONITORING_IMAGE)
    
    print('EL OUT ES: '+ str(out))
    print('EL ERR ES: '+ str(err))


###################################################################################
#============================== predict PIPELINE =================================#
###################################################################################
@pipeline(name = config["PIPELINE_METADATA"]["key"]+"-"+config["PIPELINE_METADATA"]["value"])
def predict_pipeline(credential_path   : str,
                     project           : str,
                     location          : str,
                     name_bucket       : str,
                     path_bucket       : str,
                     labels            : Dict,
                     prod_config       : Dict,
                     model_name        : str,
                     output_bq_project : str,
                     output_bq_dataset : str,
                     output_bq_table   : str):
    
    get_data_op = get_data(credential_path = credential_path,
                          name_bucket       = name_bucket,
                          path_bucket       = path_bucket)\
    .set_cpu_limit('16')\
    .set_memory_limit('64G')\
    .set_display_name('Get Data')
    """
    predict_model_endpoint_op = predict_model_endpoint(credential_path = credential_path,
                                                       data_input      = get_data_op.outputs['dataset'],
                                                       model_name      = model_name)\
    .set_cpu_limit('16')\
    .set_memory_limit('64G')\
    .set_display_name('Predict with endpoint')
    
    save_predictions_op = save_predictions(credential_path   = credential_path,
                                           data_input        = get_data_op.outputs['dataset'],
                                           predictions       = predict_model_endpoint_op.outputs["Output"],
                                           output_bq_project = output_bq_project,
                                           output_bq_dataset = output_bq_dataset,
                                           output_bq_table   = output_bq_table)\
    .set_cpu_limit('16')\
    .set_memory_limit('64G')\
    .set_display_name('Save predictions')
    """
    batch_predict_op = batch_predict(credential_path   = credential_path,
                                     project           = project,
                                     location          = location,
                                     name_bucket       = name_bucket,
                                     path_bucket       = path_bucket,
                                     labels            = labels,
                                     prod_config       = prod_config,
                                     model_name        = model_name,
                                     features          = get_data_op.outputs["features"],
                                     target            = get_data_op.outputs["target"],
                                     train_data_path   = get_data_op.outputs["train_data_path"],
                                     data_input        = get_data_op.outputs['dataset'])\
    .set_cpu_limit('16')\
    .set_memory_limit('64G')\
    .set_display_name('Batch predictions')
    
    """
    save_stats_op = save_stats(credential_path  = credential_path,
                               project          = project,
                               location         = location,
                               name_bucket      = name_bucket,
                               path_bucket      = path_bucket,
                               labels           = labels,
                               job_name         = batch_predict_op.outputs["job_name"])\
    .set_cpu_limit('16')\
    .set_memory_limit('64G')\
    .set_display_name('Save Stats Predictions')
    """

###################################################################################
#================================= COMPILE & RUN =================================#
###################################################################################

def compile_pipeline(path_bucket       : str = PATH_BUCKET, 
                     name_bucket       : str = NAME_BUCKET, 
                     compile_name_file : str = COMPILE_NAME_FILE,
                     credential_path   : str = CREDENTIAL_PATH) -> str:
    
    compiler.Compiler().compile(pipeline_func = predict_pipeline,
                                package_path  = compile_name_file)
    
    # Initialize credentials
    credentials = Credentials.from_service_account_file(credential_path)
    client = storage.Client(credentials=credentials)
    
    ### Send file(s) to bucket
    #client = storage.Client()
    bucket = client.get_bucket(name_bucket)
    blob   = bucket.blob(path_bucket+'/'+compile_name_file)
    blob.upload_from_filename('./'+compile_name_file)
    
    return f"-- OK: COMPILE -- | PATH: {path_bucket+'/'+compile_name_file}"

def run_pipeline(credential_path   : str = CREDENTIAL_PATH,
                 project           : str = PROJECT,
                 location          : str = REGION,
                 labels            : Dict = LABELS,
                 prod_config       : Dict = PROD_CONFIG,
                 model_name        : str = MODEL_NAME,
                 path_bucket       : str = PATH_BUCKET, 
                 name_bucket       : str = NAME_BUCKET, 
                 compile_name_file : str = COMPILE_NAME_FILE,
                 display_name      : str = DISPLAY_NAME,
                 output_bq_project : str = OUTPUT_BQ_PROJECT,
                 output_bq_dataset : str = OUTPUT_BQ_DATASET,
                 output_bq_table   : str = OUTPUT_BQ_TABLE) -> str:
    
    ### Parameters for pipeline job
    pipeline_parameters = dict(credential_path   = credential_path,
                               model_name        = model_name,
                               project           = project,
                               location          = location,
                               name_bucket       = name_bucket,
                               path_bucket       = path_bucket,
                               labels            = labels,
                               prod_config       = prod_config,
                               output_bq_project = output_bq_project,
                               output_bq_dataset = output_bq_dataset,
                               output_bq_table   = output_bq_table)
    
    start_pipeline = aiplatform.PipelineJob(display_name     = list(labels.values())[0],
                                            template_path    = 'gs://'+name_bucket+'/'+path_bucket+'/'+compile_name_file,
                                            pipeline_root    = 'gs://'+name_bucket+'/'+path_bucket,
                                            job_id           = display_name,
                                            labels           = labels,
                                            enable_caching   = False,
                                            location         = location,
                                            credentials      = credentials,
                                            parameter_values = pipeline_parameters)
    
    # Open the file for reading
    with open(credential_path, 'r') as file:
        credentials_json = json.load(file)

    # Access to service_account
    service_acc = credentials_json.get('client_email')

    start_pipeline.submit(service_account=service_acc)
    #start_pipeline.submit()
    
    return '-- OK RUN --'