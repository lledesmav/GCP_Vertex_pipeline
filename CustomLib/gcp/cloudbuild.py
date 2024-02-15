from typing import Tuple, Any
import subprocess
import os
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

def submit(image_name:str,
           code_path:str,
           name_bucket:str,
           project_id:str = '',
           credential_path:str = '') -> Tuple[bytes, bytes]:
    """
    Submits a Google Cloud Build job to build a Docker image from the provided code path.
    
    Args:
        image_name (str): Name of the Docker image to build.
        code_path (str): Path to the code directory containing the Dockerfile.
        project_id (str, optional): Google Cloud Project ID. Defaults to an empty string.
        credential_path (str, optional): Path to the JSON file containing the Google Cloud credentials. Defaults to an empty string.
        
    Return:
        out (bytes): Standard output from the gcloud command execution.
        err (bytes): Standard error output from the gcloud command execution.
    """
    #log_lib("Ferreylib.gcp.cloudbuild.submit")
    #Review credentials
    review_credentials(credential_path, project_id)
    # Set the Google Cloud Project ID
    env = os.environ.copy()
    _ = subprocess.run('gcloud config set project {}'.format(os.environ['GOOGLE_CLOUD_PROJECT']), shell=True, env=env)
    # Activate the service account 
    activate_command = 'gcloud auth activate-service-account --key-file={}'.format(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    _ = subprocess.run(activate_command, shell=True, env=env)
    #Define job
    command = 'gcloud builds submit --tag {} {} --gcs-source-staging-dir gs://{}/cloud-build-pipelines'.format(image_name, code_path, name_bucket)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    out = result.stdout
    err = result.stderr

    return out, err
    

