# core/aws_client.py
"""
Centralized AWS client factory to ensure proper credential handling.
This module creates AWS clients with explicit credential configuration.
"""
import boto3
from botocore.config import Config
from core.config import settings
from core.logger import logger
import os


def get_bedrock_runtime_client():
    """Get Bedrock Runtime client with proper credentials."""
    try:
        # Get credentials from settings (which loads from .env) or environment
        aws_access_key_id = getattr(settings, 'AWS_ACCESS_KEY_ID', None) or os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = getattr(settings, 'AWS_SECRET_ACCESS_KEY', None) or os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session_token = getattr(settings, 'AWS_SESSION_TOKEN', None) or os.getenv('AWS_SESSION_TOKEN')
        
        client = boto3.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token  # Optional for temporary credentials
        )
        logger.info("Bedrock Runtime client initialized with credentials")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock Runtime client: {str(e)}")
        raise


def get_bedrock_agent_client():
    """Get Bedrock Agent Runtime client with proper credentials."""
    try:
        config = Config(
            read_timeout=900,
            connect_timeout=60,
            retries={'max_attempts': 0}
        )
        
        # Get credentials from settings (which loads from .env) or environment
        aws_access_key_id = getattr(settings, 'AWS_ACCESS_KEY_ID', None) or os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = getattr(settings, 'AWS_SECRET_ACCESS_KEY', None) or os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session_token = getattr(settings, 'AWS_SESSION_TOKEN', None) or os.getenv('AWS_SESSION_TOKEN')
        
        client = boto3.client(
            "bedrock-agent-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,  # Optional for temporary credentials
            config=config
        )
        logger.info("Bedrock Agent Runtime client initialized with credentials")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock Agent Runtime client: {str(e)}")
        raise


def get_s3_client():
    """Get S3 client with proper credentials."""
    try:
        # Get credentials from settings (which loads from .env) or environment
        aws_access_key_id = getattr(settings, 'AWS_ACCESS_KEY_ID', None) or os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = getattr(settings, 'AWS_SECRET_ACCESS_KEY', None) or os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session_token = getattr(settings, 'AWS_SESSION_TOKEN', None) or os.getenv('AWS_SESSION_TOKEN')
        
        client = boto3.client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token  # Optional for temporary credentials
        )
        logger.info("S3 client initialized with credentials")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {str(e)}")
        raise


def get_sqs_client():
    """Get SQS client with proper credentials."""
    try:
        # Get credentials from settings (which loads from .env) or environment
        aws_access_key_id = getattr(settings, 'AWS_ACCESS_KEY_ID', None) or os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = getattr(settings, 'AWS_SECRET_ACCESS_KEY', None) or os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session_token = getattr(settings, 'AWS_SESSION_TOKEN', None) or os.getenv('AWS_SESSION_TOKEN')
        
        client = boto3.client(
            "sqs",
            region_name=settings.SQS_REGION,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token  # Optional for temporary credentials
        )
        logger.info("SQS client initialized with credentials")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize SQS client: {str(e)}")
        raise


def get_bedrock_client():
    """Get Bedrock client with proper credentials."""
    try:
        # Get credentials from settings (which loads from .env) or environment
        aws_access_key_id = getattr(settings, 'AWS_ACCESS_KEY_ID', None) or os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = getattr(settings, 'AWS_SECRET_ACCESS_KEY', None) or os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session_token = getattr(settings, 'AWS_SESSION_TOKEN', None) or os.getenv('AWS_SESSION_TOKEN')
        
        client = boto3.client(
            "bedrock",
            region_name=settings.AWS_REGION,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token  # Optional for temporary credentials
        )
        logger.info("Bedrock client initialized with credentials")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock client: {str(e)}")
        raise


def validate_aws_credentials():
    """Validate that AWS credentials are properly configured."""
    # Check both settings and environment variables
    aws_access_key_id = getattr(settings, 'AWS_ACCESS_KEY_ID', None) or os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = getattr(settings, 'AWS_SECRET_ACCESS_KEY', None) or os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if not aws_access_key_id or not aws_secret_access_key:
        logger.warning("Missing AWS credentials in both settings and environment variables")
        logger.info("AWS credentials not found. Make sure to set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env file, "
                   "or configure AWS CLI with 'aws configure'")
        return False
    
    logger.info("AWS credentials found and validated")
    return True