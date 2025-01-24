"""MLCube handler file"""
import os 
os.environ["MKL_SERVICE_FORCE_INTEL"] = '1'  # see issue #152
import typer
import os
from flask import Flask, request, jsonify
import json
import boto3
import os
from urllib.parse import urlparse
from os import makedirs
import sys

TASK_NAME = "BraTS-PED"
if TASK_NAME == "BraTS-SSA":
    import runner_ssa as runner
elif TASK_NAME == "BraTS-PED":
    import runner_ped as runner

app = typer.Typer()
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def get_last_folder(s3_prefix):

    s3_prefix = s3_prefix.rstrip('/')
    folders = s3_prefix.split('/')
    if folders:
        return folders[-1]
    else:
        return ''

def parse_s3_url(s3_url):
    parsed_url = urlparse(s3_url)
    if parsed_url.scheme != 's3':
        raise ValueError(f"Not a valid S3 URL: {s3_url}")
    
    bucket = parsed_url.netloc
    key = parsed_url.path.lstrip('/')
    return bucket, key

def download_s3_prefix(s3_url, local_dir):
    
    bucket, prefix = parse_s3_url(s3_url)
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    
    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)
    
    # Iterate through all objects with the given prefix
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            # Get the relative path of the file
            relative_path = os.path.relpath(obj['Key'], prefix)
            # Construct the full local path
            local_file_path = os.path.join(local_dir, relative_path)
            # Ensure the directory exists
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            # Download the file
            s3.download_file(bucket, obj['Key'], local_file_path)
            print(f"Downloaded {obj['Key']} to {local_file_path}")

def upload_folder_to_s3(local_folder,  s3_url):
    
    bucket_name, s3_prefix = parse_s3_url(s3_url)
    s3 = boto3.client('s3')
    
    # Ensure the S3 prefix ends with a slash
    if not s3_prefix.endswith('/'):
        s3_prefix += '/'
    
    # Walk through the local folder
    for root, dirs, files in os.walk(local_folder):
        for filename in files:
            # Construct the full local path
            local_path = os.path.join(root, filename)
            
            # Construct the full S3 key
            relative_path = os.path.relpath(local_path, local_folder)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            
            # Upload the file
            try:
                s3.upload_file(local_path, bucket_name, s3_key)
                print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_key}")
            except Exception as e:
                print(f"Error uploading {local_path}: {str(e)}")

app = typer.Typer(no_args_is_help=True)

@app.command("infer")
def infer(
    data_path: str = typer.Option(..., "--data_path"),
    output_path: str = typer.Option(..., "--output_path"),
):

    #runner.setup_model_weights()
    runner.batch_processor(data_path, output_path)
    return output_path

@app.command("install")
def install():
    runner.setup_model_weights()

@app.command("hotfix")
def hotfix():
    # NOOP command for typer to behave correctly. DO NOT REMOVE OR MODIFY
    pass

help_msg = """
Use this mode to run the model with SageMaker Endpoint asynchronous invocation.
The model will download the input data from S3 directly and output the results to S3 as well.
The asynchronous call needs to be provided with input job JSON file with the below structure:


{\r\n
     "input_location" : s3://[S3_BUCKET_NAME]/[INPUT_DATA_FOLDER],\r\n
     "output_location" : S3://[S3_BUCKET_NAME]/[OUTPUT_DATA_FOLDER],\r\n
     "job_id" : [ARBITRARY_JO_IB_UNIQUE_PER_MODEL_INVOCATION]\r\n
}\r\n

Refer to SageMaker endpoint ASync invocation documentation for more information : https://docs.aws.amazon.com/sagemaker/latest/dg/async-inference-invoke-endpoint.html
"""
@app.command("endpoint", help=help_msg)
def endpoint():
    app_http = Flask(__name__)


    @app_http.route('/ping', methods=['GET'])
    def ping():
        # Health check
        return "", 200

    @app_http.route('/invocations', methods=['POST'])
    def invoke():
        import torch
        eprint("CUDA Version:")
        eprint(torch.version.cuda)

        try:
            eprint(request)
            paylaod = request.get_data().decode('utf-8')
        # paylaod = json.loads(paylaod)
            eprint(paylaod)
            eprint("[mlcube:invoke] - prediction start")
            input_path = request.json.get('input_location')
            output_path = request.json.get('output_location')
            job_id = request.json.get('job_id')
            # using boto3 , download the content of the S3 path from s3  in the sagemaker container local drive
            input_data = f"/tmp/input/{job_id}/"
            output_data = f"/tmp/output/{job_id}/"
            os.makedirs(input_data, exist_ok=True)
            os.makedirs(output_data, exist_ok=True)
            tmp_dir = input_data+"/"+get_last_folder(input_path)
            eprint(tmp_dir)
            download_s3_prefix(input_path, tmp_dir)
            import runner_ped as runner
            runner.setup_model_weights()
            runner.batch_processor("/tmp/input/", output_data)  
            eprint(f"Prediction files to export to {output_path}:")
            for entry in os.listdir(output_data):
                print(entry)
            upload_folder_to_s3(output_data, output_path)
            eprint("[mlcube:invoke] - prediction done")
            return jsonify({"OK"}), 200
        except Exception as e:
            eprint(e)
            return jsonify({"error": "bug: "+str(e)}), 501
    app_http.run(host="0.0.0.0", port=8080)

if __name__ == "__main__":
    app()

       

            
     