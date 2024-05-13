from kfp.dsl import pipeline, component, Dataset, InputPath, OutputPath, Model, Input, Output, Metrics, Artifact, Condition, ClassificationMetrics
from typing import Dict, Optional, Sequence, NamedTuple, List, Union, Tuple
from kfp import components
from kfp import compiler
from google.cloud import storage
from google.cloud import aiplatform
from datetime import datetime
from google.oauth2.service_account import Credentials
from google_cloud_pipeline_components.v1.vertex_notification_email import VertexNotificationEmailOp
from kfp import dsl

########################################################################################
#=========================  Get VARIABLES & BASE_IMAGE   ==============================#
########################################################################################
import json
import os

file                      = open('config.json')
config                    = json.load(file)

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
                                                                 'repo-'+BASE_CONTAINER_IMAGE_NAME, 
                                                                 BASE_CONTAINER_IMAGE_NAME)+'latest'


##############################################################################################
#===================================== get_data COMPONENT ===================================#
##############################################################################################
@component(base_image = BASE_IMAGE)
def get_data(dataset : OutputPath("Dataset")):
    
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
def split_data(data_input      : InputPath("Dataset"),
               data_train      : OutputPath("Dataset"),
               data_test       : OutputPath("Dataset"),):
    
    #==== Read input data from GCS ====#
    from CustomLib.gcp import cloudstorage
    data = cloudstorage.read_csv_as_df(gcs_path = data_input + '.csv')
    
    #==== Preprocessing the data ====#
    from src.utils import split_data
    train_df, test_df = split_data(data)
    
    #==== Save the df in GCS ====#
    cloudstorage.write_csv(df       = train_df, 
                           gcs_path = data_train + '.csv')
    cloudstorage.write_csv(df       = test_df, 
                           gcs_path = data_test + '.csv')
    
##############################################################################################
#=================================== train_model COMPONENT ==================================#
##############################################################################################
@component(base_image = BASE_IMAGE)
def train_model(name_bucket     : str,
                path_bucket     : str,
                data_train      : InputPath("Dataset"),
                model           : Output[Model],
                metrics         : Output[Metrics]):
    
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
    from src.utils import preprocess_data, train_model
    train_df, scaler = preprocess_data(train_df)

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
    
    #==== Save the model and scaler as a .pkl file ====#
    cloudstorage.write_pickle(model    = model_iris, 
                              gcs_path = model.path + '/model.pkl')
    cloudstorage.write_pickle(model    = scaler, 
                              gcs_path = model.path + '/scaler.pkl')
    
    
##############################################################################################
#================================ testing_model COMPONENT ===================================#
##############################################################################################
@component(base_image = BASE_IMAGE)
def testing_model(_model           : Input[Model],
                  data_test       : InputPath("Dataset"),
                  metrics         : Output[Metrics])->float:

    #==== Read test data from GCS ====#
    from CustomLib.gcp import cloudstorage
    test_df = cloudstorage.read_csv_as_df(gcs_path = data_test + '.csv')
    
    #==== Read model artifact ====#
    model = cloudstorage.read_pickle(gcs_path = _model.path + '/model.pkl')
    scaler = cloudstorage.read_pickle(gcs_path = _model.path + '/scaler.pkl')
    
    #==== Evaluating the model ====#
    from src.utils import scaler_data
    test_df = scaler_data(scaler, test_df)
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
           packages_to_install = ["google-cloud-aiplatform[prediction]==1.51.0"])
def create_custom_predict(project         : str,
                          location        : str,
                          name_bucket     : str,
                          labels          : Dict)-> str:
    
    #==== Define BASE_IMAGE ====#
    repo_name  = 'repo-'+list(labels.values())[0]
    container  = list(labels.values())[0]+'-cust-pred'
    BASE_IMAGE = '{}-docker.pkg.dev/{}/{}/{}:latest'.format(location, project, repo_name, container)
    
    #==== Build the Custom Predict Routine ====#
    from pipeline.prod_modules import build_custom_predict_routine_image
    out, err = build_custom_predict_routine_image(BASE_IMAGE             = BASE_IMAGE,
                                                  CUST_PRED_ROUTINE_PATH = "pipeline/custom_prediction",
                                                  PROJECT_ID             = project,
                                                  NAME_BUCKET            = name_bucket)
    
    print('El out es: ' + str(out))
    print('El err es: ' + str(err))
    
    return BASE_IMAGE
    
######################################################################################### 
#========================= upload_to_model_registry COMPONENT ==========================#
######################################################################################### 
@component(base_image="python:3.9",
           packages_to_install=["google-cloud-aiplatform==1.51.0", 
                                "google-auth==2.29.0",
                                "google-auth-oauthlib==1.2.0",
                                "google-auth-httplib2==0.2.0",
                                "google-api-python-client==1.12.11"])
def upload_to_model_registry(project                     : str,
                             location                    : str,
                             name_bucket                 : str,
                             path_bucket                 : str,
                             model_name                  : str,
                             serving_container_image_uri : str,
                             input_model                 : Input[Model],
                             credential_dict             : Dict = None,
                             description                 : str  = None,
                             labels                      : Dict = None,)->str:
    """
    Upload a trained model to the Google Cloud AI Platform's Model Registry, creates a version of the model, and returns the uploaded model's name.

    Args:
        project (str)                     : The Google Cloud Project ID.
        location (str)                    : The region for the Google Cloud Project.
        model_name (str)                  : The name for the model in the model registry.
        serving_container_image_uri (str) : The URI of the serving container image.
        description (str, optional)       : A description for the uploaded model. Defaults to None.
        labels (Dict, optional)           : A dictionary containing labels for the uploaded model. Defaults to None.
        credential_dict (Dict)            : A dictionary containing the service account credentials.
        input_model (Input[Model])        : The trained model to be uploaded.
    
    Returns:
        (str): The name of the uploaded model in the Google Cloud AI Platform's Model Registry.
    """
    #=== Get the correct artifact_uri for the model ===#
    artifact_uri = input_model.path+'/'
    
    #=== Generate a timestamp ===#
    from datetime import datetime
    timestamp =datetime.now().strftime("%Y%m%d%H%M%S")
    
    #=== Initialize the aiplatform with the credentials ===#
    from google.cloud import aiplatform
    from google.oauth2 import service_account
    from google.auth import default
    
    if credential_dict:
        # Initialize AI Platform with service account credentials from dict
        credentials = service_account.Credentials.from_service_account_info(credential_dict)
    else:
        # Use default credentials
        credentials, _ = default()

    aiplatform.init(project=project, location=location, credentials=credentials)
    
    #=== Check if exist a previous version of the model ===#
    model_list=aiplatform.Model.list(filter = 'display_name="{}"'.format(model_name))
    if len(model_list)>0:
        parent_model_name = model_list[0].name
    else:
        parent_model_name = None

    staging_bucket = f"gs://{name_bucket}/{path_bucket}/model-registry/{model_name}-{timestamp}"
    #=== Upload the model to Model Registry ===#
    model = aiplatform.Model.upload(display_name                    = model_name,
                                    artifact_uri                    = artifact_uri,
                                    parent_model                    = parent_model_name,
                                    description                     = description,
                                    labels                          = labels,
                                    serving_container_image_uri     = serving_container_image_uri,
                                    version_aliases                 = [model_name+'-'+timestamp],        
                                    serving_container_health_route  = "/v1/models",
                                    serving_container_predict_route = "/v1/models/predict",
                                    staging_bucket                  = staging_bucket)

    model.wait()
    
    import time
    time.sleep(300)
    
    return model.name
    
    
######################################################################################### 
#========================== deploy_model_endpoint COMPONENT ============================#
######################################################################################### 
@component(base_image="python:3.9",
           packages_to_install=["google-cloud-aiplatform==1.51.0", 
                                "google-auth==2.29.0",
                                "google-auth-oauthlib==1.2.0",
                                "google-auth-httplib2==0.2.0",
                                "google-api-python-client==1.12.11"])
def deploy_model_endpoint(project         : str,
                          location        : str,
                          model_name      : str,
                          model_id        : str,
                          machine_type    : str,
                          credential_dict : Dict = None,
                          service_account_mail : str = None,
                          labels          : Dict = None)->str:
    
    #=== Initialize the aiplatform with the credentials ===#
    from google.cloud import aiplatform
    from google.oauth2 import service_account
    from google.auth import default
    
    if credential_dict:
        # Initialize AI Platform with service account credentials from dict
        credentials = service_account.Credentials.from_service_account_info(credential_dict)
    else:
        # Use default credentials
        credentials, _ = default()
        
    aiplatform.init(project     = project, 
                    location    = location, 
                    credentials = credentials)
    
    # Request if exist a endpoint
    def get_endpoint_by_display_name(display_name: str):
        endpoints = aiplatform.Endpoint.list() # List all available endpoints
        # Filter the endpoints by the given display_name
        for endpoint in endpoints:
            if endpoint.display_name == display_name:
                return endpoint
        return None
                                                                  
    endpoint = get_endpoint_by_display_name(display_name = model_name + "-endpoint")
    if endpoint:
        pass
    else:
        endpoint = aiplatform.Endpoint.create(display_name = model_name + "-endpoint",
                                              labels       = labels)
    
    # Get the model
    model= aiplatform.Model(model_name=model_id)
    
    # Deploy the model in a endpoint
    deploy=model.deploy(endpoint=endpoint,
                        traffic_percentage=100,
                        machine_type=machine_type,
                        service_account=service_account_mail)
    
    return str(endpoint.name)
    
###################################################################################
#================================ train PIPELINE =================================#
###################################################################################
@pipeline(name = config["PIPELINE_METADATA"]["key"]+"-"+config["PIPELINE_METADATA"]["value"])
def train_pipeline(project           : str,
                   location          : str,
                   name_bucket       : str,
                   path_bucket       : str,
                   labels            : Dict,
                   model_name        : str,
                   model_description : str):
    
    notify_email_task = VertexNotificationEmailOp(recipients=config['PIPELINE_ALERT_MAILS'])

    with dsl.ExitHandler(notify_email_task):
    
        get_data_op = get_data()\
            .set_cpu_limit('1')\
            .set_memory_limit('4G')\
            .set_display_name('Get Data')

        split_data_op = split_data(data_input = get_data_op.outputs['dataset'])\
        .set_cpu_limit('1')\
        .set_memory_limit('4G')\
        .set_display_name('Split Data')

        train_model_op = train_model(name_bucket = name_bucket,
                                     path_bucket = path_bucket,
                                     data_train  = split_data_op.outputs['data_train'])\
        .set_cpu_limit('1')\
        .set_memory_limit('4G')\
        .set_display_name('Train Model')

        testing_model_op = testing_model(_model     = train_model_op.outputs['model'],
                                         data_test = split_data_op.outputs['data_test'])\
        .set_cpu_limit('1')\
        .set_memory_limit('4G')\
        .set_display_name('Testing Model')

        with Condition(testing_model_op.outputs["Output"] > 0.96, name='model-upload-condition'):
            custom_predict_op = create_custom_predict(project     = project,
                                                      location    = location,
                                                      name_bucket = name_bucket,
                                                      labels      = labels)\
            .set_cpu_limit('1')\
            .set_memory_limit('4G')\
            .set_display_name('Create Custom Predict Image')

            upload_model_op = upload_to_model_registry(project                     = project,
                                                       location                    = location,
                                                       name_bucket                 = name_bucket,
                                                       path_bucket                 = path_bucket,
                                                       model_name                  = model_name,
                                                       serving_container_image_uri = custom_predict_op.outputs["Output"],
                                                       input_model                 = train_model_op.outputs['model'],
                                                       description                 = model_description,
                                                       labels                      = labels)\
            .set_cpu_limit('1')\
            .set_memory_limit('4G')\
            .set_display_name('Save Model')

            # deploy_model_endpoint_op = deploy_model_endpoint(project         = project,
            #                                              location        = location,
            #                                              model_name      = model_name,
            #                                              model_id        = upload_model_op.outputs["Output"],
            #                                              machine_type    = 'n1-standard-2',
            #                                              labels          = labels)\
            # .set_cpu_limit('4')\
            # .set_memory_limit('8G')\
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
    client = storage.Client()
    
    ### Send file(s) to bucket
    #client = storage.Client()
    bucket = client.get_bucket(name_bucket)
    blob   = bucket.blob(path_bucket+'/'+compile_name_file)
    blob.upload_from_filename('./'+compile_name_file)
    
    return f"-- OK: COMPILE -- | PATH: {path_bucket+'/'+compile_name_file}"

def run_pipeline(project           : str = PROJECT,
                 location          : str = REGION,
                 labels            : Dict = LABELS,
                 model_name        : str = MODEL_NAME,
                 model_description : str = MODEL_DESCRIPTION,
                 path_bucket       : str = PATH_BUCKET, 
                 name_bucket       : str = NAME_BUCKET, 
                 compile_name_file : str = COMPILE_NAME_FILE,
                 display_name      : str = DISPLAY_NAME) -> str:
    
    ### Parameters for pipeline job
    pipeline_parameters = dict(project           = project,
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
                                            parameter_values = pipeline_parameters)
    
    start_pipeline.submit()
    
    return '-- OK RUN --'