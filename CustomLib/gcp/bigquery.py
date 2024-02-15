import pandas as pd
from google.cloud import bigquery
import os
from google.cloud import exceptions
#from FerreyLib.utils_log import log_lib
from typing import List, Dict, Optional
from google.oauth2 import service_account

def read_table_as_df(query:str,
                     credential_path:str = '') -> pd.DataFrame:
    """
    Reads a table from Google BigQuery into a pandas DataFrame using a SQL query.
    
    Args:
        query (str): The SQL query to fetch the table data.
        credential_path (str, optional): Path to the JSON file containing the service account key. Default is an empty string, which uses the default credentials.
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing the table data.
    """
    #log_lib("Ferreylib.gcp.bigquery.read_table_as_df")
    if credential_path != '':
        credentials = service_account.Credentials.from_service_account_file(credential_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],)
        # Create a BigQuery client
        client = bigquery.Client(project=credentials.project_id, credentials=credentials)
    else:
        # Create a BigQuery client
        client = bigquery.Client()
    # Run a query to read the entire table
    query_job = client.query(query)
    # Read the results into a DataFrame
    df = query_job.to_dataframe()

    return df

def write_df(df: pd.DataFrame,
             project_id: str, 
             dataset_id: str, 
             table_id: str, 
             credential_path: str = '', 
             schema_dict: Optional[Dict[str, List[str]]] = None, 
             if_exists: str = 'fail') -> None:
    """
    Writes a pandas DataFrame to a Google BigQuery table.
    
    Args:
        df (pd.DataFrame): The DataFrame to be written to BigQuery.
        project_id (str): The ID of the Google Cloud project.
        dataset_id (str): The ID of the dataset in BigQuery.
        table_id (str): The ID of the table in BigQuery.
        credential_path (str, optional): Path to the JSON file containing the service account key. Default is an empty string, which uses the default credentials.
        schema_dict (Dict[str, List[str]], optional): The schema of the table, provided as a dictionary. Default is None, which infers the schema from the DataFrame.
        if_exists (str, optional): What to do if the table already exists. Options are 'fail', 'replace', and 'append'. Default is 'fail'.
        
    Returns:
        None
    """
    # Authentication
    if credential_path:
        credentials = service_account.Credentials.from_service_account_file(credential_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        client = bigquery.Client(project=project_id, credentials=credentials)
    else:
        client = bigquery.Client(project=project_id)

    dataset_ref = client.dataset(dataset_id)
    
    # Check if dataset exists, if not create it
    try:
        client.get_dataset(dataset_ref)
    except exceptions.NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        client.create_dataset(dataset)
    
    table_ref = dataset_ref.table(table_id)
    
    # Check if table exists
    try:
        table = client.get_table(table_ref)
        table_exists = True
    except exceptions.NotFound:
        table_exists = False

    # Handle 'if_exists' parameter
    if if_exists == 'fail' and table_exists:
        raise ValueError(f"Table {table_id} already exists.")
        
    elif if_exists == 'replace' and table_exists:
        client.delete_table(table_ref)
        table_exists = False
    
    elif if_exists not in ['fail', 'replace', 'append']:
        raise ValueError(f"Invalid value for 'if_exists': {if_exists}")

    # Create or update table schema
    if not table_exists and schema_dict:
        schema = [bigquery.SchemaField(name, type_, mode=mode, description=desc)
                  for name, type_, mode, desc in zip(schema_dict['field_name'],
                                                     schema_dict['type'],
                                                     schema_dict['mode'],
                                                     schema_dict['description'])]
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)
             
    elif table_exists and if_exists != 'append' and schema_dict:
        new_schema = [bigquery.SchemaField(field.name, field.field_type, description=desc)
                      for field, desc in zip(table.schema, schema_dict['description'])]
        table.schema = new_schema
        client.update_table(table, ['schema'])
        
    # Write DataFrame to BigQuery
    job_config = bigquery.LoadJobConfig()
    if schema_dict:
        job_config.schema = [bigquery.SchemaField(name, type_, mode=mode)
                             for name, type_, mode in zip(schema_dict['field_name'],
                                                          schema_dict['type'],
                                                          schema_dict['mode'])]
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()  # Wait for job to complete
    
    #print(f"Tabla creada: {project_id}.{dataset_id}.{table_id}")

def execute_query(query: str, credential_path: str = '') -> str:
    """
    Executes a SQL query on Google BigQuery. If an error occurs, returns an error message as a string.
    
    Args:
        query (str): The SQL query to be executed.
        credential_path (str, optional): Path to the JSON file containing the service account key. Default is an empty string, which uses the default credentials.
        
    Returns:
        str: Empty string if successful, or error message if an error occurs.
    """
    try:
        # If credential_path is provided, use it to authenticate
        if credential_path != '':
            credentials = service_account.Credentials.from_service_account_file(
                credential_path, 
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            # Create a BigQuery client
            client = bigquery.Client(project=credentials.project_id, credentials=credentials)
        else:
            # Create a BigQuery client
            client = bigquery.Client()

        # Execute the query
        query_job = client.query(query)
        # Wait for the query to finish
        query_job.result()
        print('Correct execution')
        
        return "Ejecución correcta"  # Return an empty string to indicate success

    except Exception as e:
        # Return the error message as a string
        print('Execute error: ' + str(e))
        return str(e)

    except bigquery.GoogleCloudError as gc_err:
        # Return Google Cloud specific errors as a string
        print('Execute error: '+ str(gc_err))
        return str(gc_err)


def create_table_raw(schema_dict, project, dataset, table_name, credential_path):
    """
    Generate SQL statement to create a BigQuery table.
    
    Parameters:
    - schema_dict: Dictionary containing column names and their data types
    - project: GCP Project name
    - dataset: BigQuery dataset name
    - table_name: BigQuery table name
    
    Returns:
    - sql: String containing the SQL statement to create the table
    """
    
    # Initialize SQL string
    sql = f"CREATE TABLE `{project}.{dataset}.{table_name}` \n(\n"
    
    # Iterate through the schema dictionary to add columns and types
    for column, data_type in schema_dict.items():
        sql += f"    {column} STRING,\n"
    
    # Remove trailing comma and add closing parenthesis
    sql = sql.rstrip(",\n") + "\n);"
    
    output = execute_query(query = sql, 
                  credential_path = credential_path)
    return output

def add_config_processed(schema_dict, config_id, domain, table_name, project, staging_dataset, destination_dataset, principal_keys, credential_path):
    """
    Generate SQL statement to insert a new record into the CNF_CDM_CONFIG table.
    """
    
    columns_str_list = []
    for col, dtype_or_expr in schema_dict.items():
        if dtype_or_expr == 'DATETIME':  # Special handling for DATETIME columns
            #column_str = f'''CASE WHEN REGEXP_CONTAINS(CAST({col} AS STRING), r"^\\\\d{{4}}-\\\\d{{2}}-\\\\d{{2}}T\\\\d{{2}}:\\\\d{{2}}:\\\\d{{2}}$") THEN CAST(TIMESTAMP_SECONDS(CAST({col} AS INT64)) AS DATETIME) ELSE NULL END AS {col}'''
            #column_str = f'''CASE WHEN REGEXP_CONTAINS({col}, r"^\\\\d{{4}}-\\\\d{{2}}-\\\\d{{2}}T\\\\d{{2}}:\\\\d{{2}}:\\\\d{{2}}$") THEN CAST(TIMESTAMP_SECONDS(CAST({col} AS INT64)) AS DATETIME) ELSE NULL END AS {col}'''
            #column_str = f'''CASE WHEN REGEXP_CONTAINS({col}, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$") THEN CAST({col} AS DATETIME) ELSE NULL END AS {col}'''
            column_str = f'''CAST({col} AS DATETIME) AS {col}'''
        elif dtype_or_expr == 'DATE':
            column_str = f'''CASE WHEN {col} = "NaT" THEN NULL ELSE CAST({col} AS DATE) END AS {col}'''
        elif dtype_or_expr == 'GEOGRAPHY':
            column_str = f'''ST_GEOGFROMTEXT({col}) AS {col}''' 
        elif dtype_or_expr == 'FLOAT':
            column_str = f"CAST({col} AS FLOAT64) AS {col}"
        elif isinstance(dtype_or_expr, str):  # Simple types
            column_str = f"CAST({col} AS {dtype_or_expr}) AS {col}" if dtype_or_expr != "STRING" else col
        else:  # Custom SQL expression
            column_str = f"{dtype_or_expr} AS {col}"
        columns_str_list.append(column_str)
    
    columns_str = ", ".join(columns_str_list)
    pk_clause = " AND ".join([f"A.{key} = B.{key}" for key in principal_keys])
    npk_clause = " OR ".join([
        f"COALESCE(A.{col},\\'\\') <> COALESCE(B.{col},\\'\\')" if dtype_or_expr == "STRING" else 
        f"CAST(A.{col} AS STRING) <> CAST(B.{col} AS STRING)" if dtype_or_expr == "DATETIME" else
        f"COALESCE(ST_ASTEXT(A.{col}),\\'\\') <> COALESCE(ST_ASTEXT(B.{col}),\\'\\')" if dtype_or_expr == "GEOGRAPHY" else
        f"A.{col} <> B.{col}" 
        for col, dtype_or_expr in schema_dict.items() if col not in principal_keys])
    
    sql = f"""INSERT INTO `{config_id}` 
    (DOMINIO, TABLA, ORIGEN, DATASET_STAGE, DATASET_DESTINO, COLUMNAS, SQL_1, PK, TRUNCATE_TABLE, NPK)
    VALUES
    (
      '{domain}',
      '{table_name}',
      '{project}',
      '{staging_dataset}',
      '{destination_dataset}',
      '{columns_str}',
      '',
      '{pk_clause}',
      'N',
      '{npk_clause}'
    );"""
    
    print(sql)
    
    output = execute_query(query = sql, 
                  credential_path = credential_path)
    return output
