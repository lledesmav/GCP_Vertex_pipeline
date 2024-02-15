import subprocess
import os
from typing import Tuple, Any
#from FerreyLib.utils_log import log_lib

def review_credentials(credential_path:str, project_id:str) -> None:
    """
    Sets environment variables for Google Application Credentials and Google Cloud Project if they are not already set.
    
    Args:
        credential_path (str): Path to the JSON file containing the Google Cloud credentials.
        project_id (str): Google Cloud Project ID.
        
    Return: 
        None
    """
    get_credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "Undefined")
    get_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "Undefined")
    if get_credentials == "Undefined":
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    if get_project == "Undefined":
        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id

def service_request(service_name:str,
                    location:str,
                    project_id:str = '',
                    credential_path:str = '') -> Tuple[bytes, bytes]:
    """
    Describes a Google Cloud Run service by running a gcloud command.
    
    Args:
        service_name (str): Name of the Cloud Run service to describe.
        location (str): The region of the service.
        project_id (str, optional): Google Cloud Project ID. Defaults to an empty string.
        credential_path (str, optional): Path to the JSON file containing the Google Cloud credentials. Defaults to an empty string.
        
    Return:
        out (bytes): Standard output from the gcloud command execution.
        err (bytes): Standard error output from the gcloud command execution.
    """
    #log_lib("Ferreylib.gcp.cloudrun.service_request")
    # Review credentials
    review_credentials(credential_path, project_id)
    # Set the Google Cloud Project ID
    env = os.environ.copy()
    _ = subprocess.run('gcloud config set project {}'.format(os.environ['GOOGLE_CLOUD_PROJECT']), shell=True, env=env)
    # Activate the service account 
    activate_command = 'gcloud auth activate-service-account --key-file={}'.format(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    _ = subprocess.run(activate_command, shell=True, env=env)
    # Define job
    command = "gcloud run services describe '{}' --platform managed --region '{}'".format(service_name, location)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    out = result.stdout
    err = result.stderr
    return out, err

def service_create(service_name:str,
                   location:str,
                   image_name:str,
                   memory:str = '1Gi',
                   max_instances:str = '5',
                   project_id:str = '',
                   credential_path:str = '') -> Tuple[bytes, bytes]:
    """
    Description: Creates a new Google Cloud Run service using the specified parameters.
    
    Args:
        service_name (str): Name of the Cloud Run service to create.
        location (str): The region of the service.
        image_name (str): Name of the Docker image to deploy.
        memory (str, optional): Memory allocated to the service. Defaults to '1Gi'.
        max_instances (str, optional): Maximum number of instances for the service. Defaults to '5'.
        project_id (str, optional): Google Cloud Project ID. Defaults to an empty string.
        credential_path (str, optional): Path to the JSON file containing the Google Cloud credentials. Defaults to an empty string.
        
    Return:
        out (bytes): Standard output from the gcloud command execution.
        err (bytes): Standard error output from the gcloud command execution.
    """
    #log_lib("Ferreylib.gcp.cloudrun.service_create")
    # Review credentials
    review_credentials(credential_path, project_id)
    # Set the Google Cloud Project ID
    env = os.environ.copy()
    _ = subprocess.run('gcloud config set project {}'.format(os.environ['GOOGLE_CLOUD_PROJECT']), shell=True, env=env)
    # Activate the service account 
    activate_command = 'gcloud auth activate-service-account --key-file={}'.format(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    _ = subprocess.run(activate_command, shell=True, env=env)
    # Define job
    command = "gcloud run deploy {} \
    --region {} \
    --image {} \
    --platform managed \
    --memory {} \
    --max-instances {} \
    --allow-unauthenticated \
    --set-env-vars AIP_HTTP_PORT=8080".format(service_name, location, image_name, memory, max_instances)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    out = result.stdout
    err = result.stderr
    return out, err