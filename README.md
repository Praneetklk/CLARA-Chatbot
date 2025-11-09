# AI-ML-Medlaunch-portal

## Current Projects
- AI Chatbot
- Survey Report Mapping
- Web Crawler and Data Mapper

# Chatbot Service
<details>
<summary>Chatbot Project Structure and steps to run it</summary>

## Running the service in Development
### Software Required
- Python 3.12
- Docker [Docker Desktop Installation](https://docs.docker.com/desktop/)

#### Note
The package manager used by the service is UV. So, to run the service in local environment, you need to install UV.
[Installation guide](https://docs.astral.sh/uv/getting-started/installation/)
    - If you have pip, you can install uv directly through ```pip install uv```

### Syncing the environment with the repo
- After installing repo, run ```uv sync``` to be in sync with the repo
- To enable the virtual environment, run
    - ```source .venv\Scripts\activate``` for windows
    - ```source .venv/bin/activate``` for Linux and MacOS
- To add packages, run `uv add <packages-name>`

### Running the Necessary Images for Development
- After Installing Docker, run the following command
    - `docker compose -f dev.docker-compose.yml up -d`
    - This runs
        - MinIO which is a S3 compatible object storage used to replicate the API to interact with S3 bucket
        - FastAPI server with configuration tweaked to run in local machine

### Running the server in dev mode
- Run the following command
    - `cd app`
    - `fastapi dev main.py`
- Note, you need to run the MinIO docker image for the server to run properly, otherwise it'll throw an error
    - Run this command to run only MinIO services: `docker compose -f dev.docker-compose.yml up minio minio-client`

### Running the server in image
- Run the following command to create the docker image:
    - `docker compose build fastapi_server`
- Run the following command to run the docker container:
    - `docker compose up -d`
- To check logs: `docker logs fastapi_server`
- To check the status of the running containers: `docker ps`


## Repo Structure
### Inside the app directory
- `experiments` directory: All the notebooks, scripts used to test out models and flow are present here
- `main.py`: Entry file for the FastAPI server
- `models.py`: Embedding and Chatbot models are initialized here
- `vec_storage.py`: Logic for creating, loading, uploading Vector Embeddings are here
- `chat.py`: Functions to get context and send the prompt to ChatBot model
</details>

# Survey Report Mapper
<details>
<summary> Details of the Report Mapper </summary>

## Running it locally
To run it locally, you need:
    - AWS Credentials with access to S3, Bedrock
- Create a `.env` file inside `./survey_report_mapping/` and add
```
ACCESS_KEY=<access-key>
SECRET_KEY=<secret-key>
IS_DEV=True
```
- Run `test_lambda_function_local.py`

## Creating the lambda function deployment
- In Linux environment run `create_lambda_zip.sh` script to create the zip file
- Upload the file to the appropriate lambda function
</details>

# Web Crawler
[See link](/web_crawler/Readme.md)
