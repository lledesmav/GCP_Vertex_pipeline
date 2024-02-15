from kfp.dsl import pipeline, component, Dataset, InputPath, OutputPath, Model, Input, Output, Metrics, Artifact, Condition, ClassificationMetrics
from typing import Dict, Optional, Sequence, NamedTuple, List, Union, Tuple
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
COMPILE_NAME_FILE = "train-"+config["PIPELINE_METADATA"]["value"]+'.yaml'
TIMESTAMP         = datetime.now().strftime("%Y%m%d%H%M%S")
DISPLAY_NAME      = config["PIPELINE_METADATA"]["key"]+"-"+config["PIPELINE_METADATA"]["value"]+'-train-{}'.format(TIMESTAMP)

BASE_CONTAINER_IMAGE_NAME = config["PIPELINE_METADATA"]["value"]
BASE_IMAGE                = '{}-docker.pkg.dev/{}/{}/{}:'.format(REGION, 
                                                                 PROJECT, 
                                                                 BASE_CONTAINER_IMAGE_NAME, 
                                                                 BASE_CONTAINER_IMAGE_NAME)+'latest'

import os
os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") == CREDENTIAL_PATH

from google.oauth2.service_account import Credentials
credentials = Credentials.from_service_account_file(CREDENTIAL_PATH)
aiplatform.init(project=PROJECT, location=REGION, credentials=credentials)

##############################################################################################
#===================================== get_data COMPONENT ===================================#
##############################################################################################
@component(base_image = BASE_IMAGE)
def get_data(credential_path : str,
             dataset         : OutputPath("Dataset")):
    
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
    
    #==== Save the df in GCS ====#
    from CustomLib.gcp import cloudstorage
    cloudstorage.write_csv(df       = data, 
                           gcs_path = dataset + '.csv')
    
    
##############################################################################################
#=================================== split_data COMPONENT ===================================#
##############################################################################################
@component(base_image = BASE_IMAGE)
def split_data(credential_path : str,
               data_input      : InputPath("Dataset"),
               scaler          : Output[Model],
               data_train      : OutputPath("Dataset"),
               data_test       : OutputPath("Dataset"),):
    
    #==== Define credentials ====#
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    
    #==== Read input data from GCS ====#
    from CustomLib.gcp import cloudstorage
    data = cloudstorage.read_csv_as_df(gcs_path = data_input + '.csv')
    
    #==== Preprocessing the data ====#
    from src.utils import preprocess_data
    train_df, test_df, scaler_model = preprocess_data(data)
    
    #==== Save the df in GCS ====#
    cloudstorage.write_csv(df       = train_df, 
                           gcs_path = data_train + '.csv')
    cloudstorage.write_csv(df       = test_df, 
                           gcs_path = data_test + '.csv')
    
    #==== Save the scaler in GCS ====#
    cloudstorage.write_pickle(model    = scaler_model, 
                              gcs_path = scaler.path + '/scaler.pkl')
    
##############################################################################################
#=================================== train_model COMPONENT ==================================#
##############################################################################################
@component(base_image = BASE_IMAGE)
def train_model(credential_path : str,
                name_bucket     : str,
                path_bucket     : str,
                data_train      : InputPath("Dataset"),
                scaler          : Input[Model],
                model           : Output[Model],
                metrics         : Output[Metrics]):

    #==== Define credentials ====#
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    
    #==== Read train data from GCS ====#
    from CustomLib.gcp import cloudstorage
    train_df = cloudstorage.read_csv_as_df(gcs_path = data_train + '.csv')
    
    #==== Save train data for monitoring ====#
    file_name = "last_training_path.txt"
    gcs_path = f"gs://{name_bucket}/{path_bucket}/{file_name}"
    with open(file_name, 'w') as file:
        file.write(data_train + '.csv')
    cloudstorage.write_text(text_path = file_name, 
                            gcs_path  = gcs_path)
    
    #==== Training the model ====#
    from src.utils import train_model
    # Separating features and target for training
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']

    model_iris = train_model(X_train, y_train)
    
    #==== Evaluating the model ====#
    predictions = model_iris.predict(X_train)
    
    #==== Getting metrics ====#
    from src.utils import get_metrics
    log, conf_matrix = get_metrics(y_train, predictions)
    
    for metric,value in log.items():
        metrics.log_metric(metric, value)
        
    #==== Read scaler artifact ====#
    scaler = cloudstorage.read_pickle(gcs_path = scaler.path + '/scaler.pkl')
    
    #==== Save the model and scaler as a .pkl file ====#
    cloudstorage.write_pickle(model    = model_iris, 
                              gcs_path = model.path + '/model.pkl')
    cloudstorage.write_pickle(model    = scaler, 
                              gcs_path = model.path + '/scaler.pkl')
    
    
##############################################################################################
#================================ testing_model COMPONENT ===================================#
##############################################################################################
@component(base_image = BASE_IMAGE)
def testing_model(credential_path : str,
                   model           : Input[Model],
                   data_test       : InputPath("Dataset"),
                   metrics         : Output[Metrics])->float:
    
    #==== Define credentials ====#
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    
    #==== Read test data from GCS ====#
    from CustomLib.gcp import cloudstorage
    test_df = cloudstorage.read_csv_as_df(gcs_path = data_test + '.csv')
    
    #==== Read model artifact ====#
    model = cloudstorage.read_pickle(gcs_path = model.path + '/model.pkl')
    
    #==== Evaluating the model ====#
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
    predictions = model.predict(X_test)
    
    #==== Getting metrics ====#
    from src.utils import get_metrics
    logs, conf_matrix = get_metrics(y_test, predictions)
    
    for metric,value in logs.items():
        metrics.log_metric(metric, value)
    
    return logs['f1']
    
#####################################################################################################
#============================ create_custom_predict_container COMPONENT ============================#
#####################################################################################################
@component(base_image          = BASE_IMAGE,
           packages_to_install = ["google-cloud-aiplatform[prediction]==1.31.0"])
def create_custom_predict(credential_path : str,
                          project         : str,
                          location        : str,
                          name_bucket     : str,
                          labels          : Dict)-> str:
    #==== Define credentials ====#
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    os.environ['GOOGLE_CLOUD_PROJECT'] = project
    
    #==== Define BASE_IMAGE ====#
    repo_name  = list(labels.values())[0]
    container  = repo_name+'-cust-pred'
    BASE_IMAGE = '{}-docker.pkg.dev/{}/{}/{}:latest'.format(location, project, repo_name, container)
    
    #==== Build the Custom Predict Routine ====#
    from pipeline.prod_modules import build_custom_predict_routine_image
    out, err = build_custom_predict_routine_image(BASE_IMAGE             = BASE_IMAGE,
                                                  CUST_PRED_ROUTINE_PATH = "pipeline/custom_prediction",
                                                  NAME_BUCKET            = name_bucket)
    
    print('El out es: ' + str(out))
    print('El err es: ' + str(err))
    
    return BASE_IMAGE
    
######################################################################################### 
#========================= upload_to_model_registry COMPONENT ==========================#
######################################################################################### 
@component(base_image = BASE_IMAGE,
           packages_to_install=["google-cloud-aiplatform==1.31.0", 
                                "google-auth==2.17.3",
                                "google-auth-oauthlib==0.4.6",
                                "google-auth-httplib2==0.1.0",
                                "google-api-python-client==1.8.0"])
def upload_to_model_registry(project                     : str,
                             location                    : str,
                             model_name                  : str,
                             serving_container_image_uri : str,
                             credential_path             : str,
                             input_model                 : Input[Model],
                             description                 : str  = None,
                             labels                      : Dict = None,)->str:
    import time
    # Sleep for 600 seconds (10 minutes)
    time.sleep(600)
    
    #=== Get the correct artifact_uri for the model ===#
    artifact_uri = input_model.path+'/'
    print('El artifact uri es  : '+str(artifact_uri))
    
    #=== Generate a timestamp ===#
    from datetime import datetime
    timestamp =datetime.now().strftime("%Y%m%d%H%M%S")
    
    #=== Initialize the aiplatform with the credentials ===#
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    
    from google.oauth2.service_account import Credentials
    credentials = Credentials.from_service_account_file(credential_path)
    
    from google.cloud import aiplatform
    aiplatform.init(project  = project, 
                    location  = location,
                    credentials=credentials)
    
    #=== Check if exist a previous version of the model ===#
    model_list=aiplatform.Model.list(filter = 'display_name="{}"'.format(model_name))
    if len(model_list)>0:
        parent_model_name = model_list[0].name
    else:
        parent_model_name = None
    
    print('El URI es: '+ input_model.path.replace("/gcs/", "gs://", 1)+'/model_registry')

    #=== Upload the model to Model Registry ===#
    model = aiplatform.Model.upload(display_name                    = model_name,
                                    artifact_uri                    = artifact_uri,
                                    parent_model                    = parent_model_name,
                                    description                     = description,
                                    labels                          = labels,
                                    serving_container_image_uri     = serving_container_image_uri,
                                    version_aliases                 = [model_name+'-'+timestamp],  
                                    staging_bucket                  = input_model.path.replace("/gcs/", "gs://", 1)+'/model_registry',
                                    serving_container_health_route  = "/v1/models",
                                    serving_container_predict_route = "/v1/models/predict")

    model.wait()
    
    return model.name
    
    
######################################################################################### 
#========================== deploy_model_endpoint COMPONENT ============================#
######################################################################################### 
@component(base_image = BASE_IMAGE)
def deploy_model_endpoint(project         : str,
                          location        : str,
                          model_name      : str,
                          model_id        : str,
                          credential_path : str)->str:
    
    #=== Initialize the aiplatform with the credentials ===#
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    
    from google.oauth2.service_account import Credentials
    credentials = Credentials.from_service_account_file(credential_path)
    
    from google.cloud import aiplatform
    aiplatform.init(project  = project, 
                    location = location,
                    credentials=credentials)
    
    # Request if exist a endpoint
    from pipeline.prod_modules import get_endpoint_by_display_name
    endpoint = get_endpoint_by_display_name(display_name = model_name + "-endpoint")
    if endpoint:
        pass
    else:
        endpoint = aiplatform.Endpoint.create(display_name = model_name + "-endpoint")
    
    # Get the model
    model= aiplatform.Model(model_name=model_id)
    
    # Deploy the model in a endpoint
    deploy=model.deploy(endpoint=endpoint,
                        min_replica_count = 1,
                        traffic_percentage=100,
                        machine_type="n1-standard-2")
    
    return str(endpoint.name)
    
###################################################################################
#================================ train PIPELINE =================================#
###################################################################################
@pipeline(name = config["PIPELINE_METADATA"]["key"]+"-"+config["PIPELINE_METADATA"]["value"])
def train_pipeline(credential_path   : str,
                   project           : str,
                   location          : str,
                   name_bucket       : str,
                   path_bucket       : str,
                   labels            : Dict,
                   model_name        : str,
                   model_description : str):
    get_data_op = get_data(credential_path = credential_path)\
                                          .set_cpu_limit('1')\
                                          .set_memory_limit('4G')\
                                          .set_display_name('Get Data')
    
    split_data_op = split_data(credential_path = credential_path,
                               data_input      = get_data_op.outputs['dataset'])\
    .set_cpu_limit('1')\
    .set_memory_limit('4G')\
    .set_display_name('Split Data')
    
    train_model_op = train_model(credential_path = credential_path,
                                 name_bucket     = name_bucket,
                                 path_bucket     = path_bucket,
                                 data_train      = split_data_op.outputs['data_train'],
                                 scaler          = split_data_op.outputs['scaler'])\
    .set_cpu_limit('1')\
    .set_memory_limit('4G')\
    .set_display_name('Train Model')
    
    testing_model_op = testing_model(credential_path = credential_path,
                                     model           = train_model_op.outputs['model'],
                                     data_test       = split_data_op.outputs['data_test'])\
    .set_cpu_limit('1')\
    .set_memory_limit('4G')\
    .set_display_name('Testing Model')
    
    with Condition(testing_model_op.outputs["Output"] > 0.96, name='model-upload-condition'):
        custom_predict_op = create_custom_predict(credential_path = credential_path,
                                                  project         = project,
                                                  location        = location,
                                                  name_bucket     = name_bucket,
                                                  labels          = labels)\
        .set_cpu_limit('1')\
        .set_memory_limit('4G')\
        .set_display_name('Create Custom Predict Image')

        upload_model_op = upload_to_model_registry(project                     = project,
                                                   location                    = location,
                                                   model_name                  = model_name,
                                                   serving_container_image_uri = custom_predict_op.outputs["Output"],
                                                   credential_path             = credential_path,
                                                   input_model                 = train_model_op.outputs['model'],
                                                   description                 = model_description,
                                                   labels                      = labels)\
        .set_cpu_limit('1')\
        .set_memory_limit('4G')\
        .set_display_name('Save Model')
        
        # deploy_model_endpoint_op = deploy_model_endpoint(project         = project,
        #                                                  location        = location,
        #                                                  model_name      = model_name,
        #                                                  model_id        = upload_model_op.outputs["Output"],
        #                                                  credential_path = credential_path)\
        # .set_cpu_limit('1')\
        # .set_memory_limit('4G')\
        # .set_display_name('Deploy model in an endpoint')

###################################################################################
#================================= COMPILE & RUN =================================#
###################################################################################

def compile_pipeline(path_bucket       : str = PATH_BUCKET, 
                     name_bucket       : str = NAME_BUCKET, 
                     compile_name_file : str = COMPILE_NAME_FILE) -> str:
    
    compiler.Compiler().compile(pipeline_func = train_pipeline,
                                package_path  = compile_name_file)
    # Initialize credentials
    credentials = Credentials.from_service_account_file(CREDENTIAL_PATH)
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
                 model_name        : str = MODEL_NAME,
                 model_description : str = MODEL_DESCRIPTION,
                 path_bucket       : str = PATH_BUCKET, 
                 name_bucket       : str = NAME_BUCKET, 
                 compile_name_file : str = COMPILE_NAME_FILE,
                 display_name      : str = DISPLAY_NAME) -> str:
    
    ### Parameters for pipeline job
    pipeline_parameters = dict(credential_path   = credential_path,
                               project           = project,
                               location          = location,
                               name_bucket       = name_bucket,
                               path_bucket       = path_bucket,
                               labels            = labels,
                               model_name        = model_name,
                               model_description = model_description)
    
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

    # Execution of the pipeline
    start_pipeline.submit(service_account=service_acc)
    
    return '-- OK RUN --'