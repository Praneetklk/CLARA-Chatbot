from core.config import settings
from core.logger import logger
from core.aws_client import get_bedrock_agent_client, validate_aws_credentials

# Validate AWS credentials on module import
validate_aws_credentials()

# Boto3 client to interact with the Bedrock Agent Runtime for the Knowledge Base
bedrock_agent_client = get_bedrock_agent_client()
logger.info("Bedrock Agent Runtime client connected.")

fallback_response = "Unable to answer your question. Please contact DNV representative (healthcare@dnv.com) for your inquiry. ".strip()  # noqa

logger.info("Models module loaded successfully")