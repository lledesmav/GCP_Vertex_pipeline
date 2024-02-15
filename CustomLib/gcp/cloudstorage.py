import pandas as pd
from google.cloud import storage
from typing import Tuple, Any
import subprocess
import io
import os
import json
#from FerreyLib.utils_log import log_lib

def split_gcs_path(gcs_path:str) -> Tuple[str, str]:
    """
    Splits the input Google Cloud Storage (GCS) path into a bucket name and a blob name.
    
    Args: 
        gcs_path (str): The GCS path to split.
        
    Returns: 
        A tuple containing the bucket name and blob name.
    """
    from urllib.parse import urlsplit
    # Split the URL components
    url_parts = urlsplit(gcs_path.replace('/gcs/','gs://')) #Replace some inconsistency
    # Extract the bucket name
    bucket_name = url_parts.netloc
    # Extract the blob_name (excluding the leading '/')
    blob_name = url_parts.path[1:]
    
    return bucket_name, blob_name
        
def write_csv(df: pd.DataFrame, 
              gcs_path:str,
              credential_path:str = '') -> None:
    """
    Writes a pandas DataFrame to a CSV file in Google Cloud Storage.
    
    Args:
        df (pd.DataFrame): The DataFrame to write.
        gcs_path (str): The GCS path to write the CSV file to.
        credential_path (str, optional): The path to the JSON file containing the Google Cloud credentials. Defaults to an empty string.
    
    Returns: 
        None
    """
    #log_lib("Ferreylib.gcp.cloudstorage.write_csv")
    if credential_path != '':
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(credential_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        # Initialize the GCS client
        storage_client = storage.Client(project=credentials.project_id, credentials=credentials)
    else:
        # Initialize the GCS client
        storage_client = storage.Client()
    #Split gcs path
    bucket_name, blob_name = split_gcs_path(gcs_path)
    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Convert the DataFrame to a CSV string
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    # Create a blob and upload the CSV data
    blob = bucket.blob(blob_name)
    blob.upload_from_string(csv_data, content_type='text/csv')
    
def read_csv_as_df(gcs_path:str,
                   credential_path:str = '') -> pd.DataFrame:
    """
    Reads a CSV file from Google Cloud Storage and returns it as a pandas DataFrame.
    
    Args:
        gcs_path (str): The GCS path of the CSV file to read.
        credential_path (str, optional): The path to the JSON file containing the Google Cloud credentials. Defaults to an empty string.
        
    Returns: 
        A pandas DataFrame containing the data from the CSV file.
    """
    #log_lib("Ferreylib.gcp.cloudstorage.read_csv_as_df")
    if credential_path != '':
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(credential_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        # Initialize the GCS client
        storage_client = storage.Client(project=credentials.project_id, credentials=credentials)
    else:
        # Initialize the GCS client
        storage_client = storage.Client()
    #Split gcs path
    bucket_name, blob_name = split_gcs_path(gcs_path)
    # Get the bucket and blob
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    # Read the CSV data as bytes
    csv_data = blob.download_as_bytes()
    # Decode the bytes to string and load it into a pandas DataFrame
    df = pd.read_csv(io.BytesIO(csv_data))

    return df

def write_pickle(model, 
                 gcs_path:str, 
                 credential_path:str = '') -> None:
    """
    Writes a Python object (such as a model) to a Pickle file in Google Cloud Storage.
    
    Args:
        model: The Python object to write.
        gcs_path (str): The GCS path to write the Pickle file to.
        credential_path (str, optional): The path to the JSON file containing the Google Cloud credentials. Defaults to an empty string.
        
    Returns: 
        None
    """
    #log_lib("Ferreylib.gcp.cloudstorage.write_pickle")
    if credential_path != '':
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(credential_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        # Initialize the GCS client
        storage_client = storage.Client(project=credentials.project_id, credentials=credentials)
    else:
        # Initialize the GCS client
        storage_client = storage.Client()
    #Split gcs path
    bucket_name, blob_name = split_gcs_path(gcs_path)
    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Convert the model to a binary pickle in-memory using pd.to_pickle()
    pkl_buffer = io.BytesIO()
    pd.to_pickle(model, pkl_buffer)
    # Upload the pickle data
    blob = bucket.blob(blob_name)
    blob.upload_from_string(pkl_buffer.getvalue(), content_type='application/octet-stream')

def read_pickle(gcs_path:str, 
                credential_path:str = '') -> Any:
    """
    Reads a Pickle file from Google Cloud Storage and returns the deserialized Python object.
    
    Args:
        gcs_path (str): The GCS path of the Pickle file to read.
        credential_path (str, optional): The path to the JSON file containing the Google Cloud credentials. Defaults to an empty string.
        
    Returns: 
        The deserialized Python object stored in the Pickle file.
    """
    #log_lib("Ferreylib.gcp.cloudstorage.read_pickle")
    if credential_path != '':
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(credential_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        # Initialize the GCS client
        storage_client = storage.Client(project=credentials.project_id, credentials=credentials)
    else:
        # Initialize the GCS client
        storage_client = storage.Client()
    #Split gcs path
    bucket_name, blob_name = split_gcs_path(gcs_path)
    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Download the pickle data
    blob = bucket.blob(blob_name)
    pkl_buffer = io.BytesIO(blob.download_as_bytes())
    # Load the pickle data into a DataFrame using pd.read_pickle()
    model = pd.read_pickle(pkl_buffer)
    
    return model    

def write_text(text_path:str, 
               gcs_path:str,
               credential_path:str = '') -> None:
    """
    Writes a text file from the local file system to Google Cloud Storage.
    
    Args:
        text_path (str): The local file system path of the text file to write.
        gcs_path (str): The GCS path to write the text file to.
        credential_path (str, optional): The path to the JSON file containing the Google Cloud credentials. Defaults to an empty string.
        
    Returns: 
        None
    """
    #log_lib("Ferreylib.gcp.cloudstorage.write_text")
    if credential_path != '':
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(credential_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        # Initialize the GCS client
        storage_client = storage.Client(project=credentials.project_id, credentials=credentials)
    else:
        # Initialize the GCS client
        storage_client = storage.Client()
    #Split gcs path
    bucket_name, blob_name = split_gcs_path(gcs_path)
    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Create a blob and upload the text
    blob = bucket.blob(blob_name)
    with open(text_path, 'rb') as f:
        blob.upload_from_file(f, content_type='text/plain')
           
def read_txt_as_str(gcs_path:str, 
                    credential_path:str = '') -> str:
    """
    Reads a text file from Google Cloud Storage and returns its contents as a string.
    
    Args:
        gcs_path (str): The GCS path of the text file to read.
        credential_path (str, optional): The path to the JSON file containing the Google Cloud credentials. Defaults to an empty string.
        
    Returns: 
        The contents of the text file as a string.
    """
    #log_lib("Ferreylib.gcp.cloudstorage.read_txt_as_str")
    if credential_path != '':
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(credential_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        # Initialize the GCS client
        storage_client = storage.Client(project=credentials.project_id, credentials=credentials)
    else:
        # Initialize the GCS client
        storage_client = storage.Client()
    #Split gcs path
    bucket_name, blob_name = split_gcs_path(gcs_path)
    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Download the text file and return its contents
    blob = bucket.blob(blob_name)
    
    return blob.download_as_text()

def write_df_as_json(df:pd.DataFrame, 
                     gcs_path:str, 
                     lines : bool = True,
                     credential_path:str = '') -> None:
    """
    Writes a pandas DataFrame to a JSON file in Google Cloud Storage.
    
    Args:
        df (pd.DataFrame): The DataFrame to write.
        gcs_path (str): The GCS path to write the JSON file to.
        credential_path (str, optional): The path to the JSON file containing the Google Cloud credentials. Defaults to an empty string.
        
    Returns: 
        None
    """
    #log_lib("Ferreylib.gcp.cloudstorage.write_df_as_json")
    if credential_path != '':
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(credential_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        # Initialize the GCS client
        storage_client = storage.Client(project=credentials.project_id, credentials=credentials)
    else:
        # Initialize the GCS client
        storage_client = storage.Client()
    #Split gcs path
    bucket_name, blob_name = split_gcs_path(gcs_path)
    # Convert the DataFrame to a JSON string
    if lines:
        json_str = df.to_json(orient='records', lines=True)
    else:
        json_str = df.to_json(orient='records')
    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Upload the JSON data to GCS
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json_str, content_type='application/json')
    
def read_json_as_df(gcs_path:str, 
                    credential_path:str = '') -> pd.DataFrame:
    #log_lib("Ferreylib.gcp.cloudstorage.read_json_as_df")
    if credential_path != '':
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(credential_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        # Initialize the GCS client
        storage_client = storage.Client(project=credentials.project_id, credentials=credentials)
    else:
        # Initialize the GCS client
        storage_client = storage.Client()
    #Split gcs path
    bucket_name, blob_name = split_gcs_path(gcs_path)
    import json
    # Get the bucket that the file resides in
    bucket = storage_client.get_bucket(bucket_name)
    # Get the blob with the given name
    blob = bucket.blob(blob_name)
    # Download the content as a string
    json_data = blob.download_as_text()
    # Use the json library to load this string as dictionary
    data = json.loads(json_data)
    # Convert the dictionary to a pandas dataframe
    df = pd.json_normalize(data)
    return df

def write_dict_as_json(dict_data, 
                       gcs_path:str,
                      credential_path:str = '') -> None:
    #log_lib("Ferreylib.gcp.cloudstorage.write_dict_as_json")
    if credential_path != '':
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(credential_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        # Initialize the GCS client
        storage_client = storage.Client(project=credentials.project_id, credentials=credentials)
    else:
        # Initialize the GCS client
        storage_client = storage.Client()
    #Split gcs path
    bucket_name, blob_name = split_gcs_path(gcs_path)
    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Convert the dictionary to a JSON string
    json_data = json.dumps(dict_data)
    # Create a blob in GCS
    blob = bucket.blob(blob_name)
    # Upload the JSON string as a blob to the bucket
    blob.upload_from_string(data=json_data,
                            content_type='application/json')

def read_json_as_dict(gcs_path, credential_path:str = ''):
    #log_lib("Ferreylib.gcp.cloudstorage.read_json_as_dict")
    if credential_path != '':
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(credential_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        # Initialize the GCS client
        storage_client = storage.Client(project=credentials.project_id, credentials=credentials)
    else:
        # Initialize the GCS client
        storage_client = storage.Client()
    #Split gcs path
    bucket_name, blob_name = split_gcs_path(gcs_path)
    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Get the blob from the bucket
    blob = bucket.blob(blob_name)
    # Download the blob's content as a string
    json_string = blob.download_as_text()
    # Convert the string to a dictionary
    dict_data = json.loads(json_string)
    return dict_data


def review_credentials(credential_path:str) -> None:
    """
    Sets the Google Cloud project and credentials if they are not already defined in the environment variables.
    
    Args:
        credential_path (str): The path to the JSON file containing the Google Cloud credentials.
        
    Returns: 
        None
    """
    get_credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "Undefined")
    if get_credentials == "Undefined":
        if credential_path != '':
            credentials = credential_path
        else:
            credentials = None
    else:
        credentials = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
        
    if credentials != None:
        flag =  f" -o 'Credentials:gs_service_key_file={credentials}' "
    else:
        flag = ""
        
    return flag

def copy_file(source:str,
              destination:str,
              credential_path:str = '') -> Tuple[bytes, bytes]:
    """
    Copies a file from the source to the destination within Google Cloud Storage.
    
    Args:
        source (str): The source file path.
        destination (str): The destination file path.
        credential_path (str, optional): The path to the JSON file containing the Google Cloud credentials. Defaults to an empty string.
        
    Returns: 
        A tuple containing the standard output and standard error from the gsutil cp command.
    """
    #log_lib("Ferreylib.gcp.cloudstorage.copy_file")
    # Review credentials
    flag = review_credentials(credential_path)
    #Define job
    command = 'gsutil ' + flag + 'cp {} {}'.format(source, destination)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = result.stdout
    err = result.stderr
    return out, err
    
def directory_list(path:str,
                   credential_path:str = '') -> Tuple[bytes, bytes]:
    """
    Lists the files in a Google Cloud Storage directory.
    
    Args:
        path (str): The path of the directory to list.
        credential_path (str, optional): The path to the JSON file containing the Google Cloud credentials. Defaults to an empty string.
        
    Returns: A tuple containing the standard output and standard error from the gsutil ls command.
    """
    #log_lib("Ferreylib.gcp.cloudstorage.directory_list")
    # Review credentials
    flag = review_credentials(credential_path)
    #Define job
    command = 'gsutil ' + flag + 'ls {}'.format(path)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = result.stdout
    err = result.stderr
    return out, err
    
def upload_file(source_local:str, destination_gcs:str, credential_path:str = '') -> None:
    """
    This function uploads a local file to a Google Cloud Storage bucket using a specified service account or default credentials if none are provided.

    Args:
        source_local (str): The local path of the file to be uploaded.
        destination_gcs (str): The destination path on Google Cloud Storage where the file will be uploaded.
        credential_path (str, optional): The path to the service account JSON key file for authentication. If not provided or empty, the function uses default credentials.
        
    Returns: None
    """
    #log_lib("Ferreylib.gcp.cloudstorage.upload_file")
    if credential_path != '':
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(credential_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        # Initialize the GCS client
        storage_client = storage.Client(project=credentials.project_id, credentials=credentials)
    else:
        # Initialize the GCS client
        storage_client = storage.Client()
    #Split gcs path
    bucket_name, blob_name = split_gcs_path(destination_gcs)
    # Get the bucket
    bucket = storage_client.bucket(bucket_name)
    # Upload file
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(source_local)
    
def upload_string(data            : str, 
                  destination_gcs : str, 
                  credential_path:str = '') -> None:
    
    if credential_path != '':
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(credential_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        # Initialize the GCS client
        storage_client = storage.Client(project=credentials.project_id, credentials=credentials)
    else:
        # Initialize the GCS client
        storage_client = storage.Client()
    #Split gcs path
    bucket_name, blob_name = split_gcs_path(destination_gcs)
    # Get the bucket
    bucket = storage_client.bucket(bucket_name)
    # Upload data
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data)
    

def check_file_exists(gcs_path:str, credential_path:str = ''):
    """
    Check if a specific file exists in a specific path in Cloud Storage.

    Parameters:
        bucket_name (str): Name of the Cloud Storage bucket.
        file_path (str): The path to the file in the Cloud Storage bucket.
        credential_path (str): Path to the credentials.json file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    #log_lib("Ferreylib.gcp.cloudstorage.write_dict_as_json")
    if credential_path != '':
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_file(credential_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        # Initialize the GCS client
        storage_client = storage.Client(project=credentials.project_id, credentials=credentials)
    else:
        # Initialize the GCS client
        storage_client = storage.Client()
    
    try:
        #Split gcs path
        bucket_name, blob_name = split_gcs_path(gcs_path)
        # Get the bucket and file
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Check if the file exists
        return blob.exists()
    
    except Exception as e:
        # Handle exceptions, such as invalid credentials or connection issues
        print(f"An error occurred: {e}")
        return False