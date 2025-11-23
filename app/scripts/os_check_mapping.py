from dotenv import load_dotenv
load_dotenv()

import json
import boto3
import requests
from requests_aws4auth import AWS4Auth

OS_ENDPOINT = "https://search-test-kb-domain-ojwpwg75h6wsb5nqxpztddob7y.us-east-1.es.amazonaws.com"
INDEX_NAME = "clara_chunks_v1"
REGION = "us-east-1"
SERVICE = "es"  # classic OpenSearch domains use "es"

session = boto3.Session()
creds = session.get_credentials()
if creds is None:
    raise RuntimeError("No AWS credentials found. Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY or AWS_PROFILE.")

frozen = creds.get_frozen_credentials()
awsauth = AWS4Auth(
    frozen.access_key,
    frozen.secret_key,
    REGION,
    SERVICE,
    session_token=frozen.token,
)

url = f"{OS_ENDPOINT}/{INDEX_NAME}/_mapping?pretty"
resp = requests.get(url, auth=awsauth)

print("Status:", resp.status_code)
print(resp.text)
