import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

# from services.storage import download_vectorstore
# from services.vec_storage import get_vec_store_handler, load_vecstore
from core.config import settings
from core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup. The old logic for loading a local vector
    store has been removed as it is no longer needed.
    """
    logger.info("Lifespan startup: Ready to serve requests.")
    yield
    logger.info("Lifespan shutdown.")

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Try to download vector store
#     # from object storage first
#     try:
#         if os.path.exists(
#             f"{settings.LOCAL_DATA_PATH}/faiss_index/index.faiss"
#         ) and os.path.exists(f"{settings.LOCAL_DATA_PATH}/faiss_index/index.pkl"):
#             load_vecstore()
#             logger.info("Loaded Local Vector Store")
#         else:
#             download_vectorstore()
#             load_vecstore()
#             logger.info("Loaded Vector Store from object store")
#     except Exception as e:
#         try:
#             logger.info(
#                 (
#                     f"When tyring to load the vector store from object store, encountered exception {str(e)}\n"
#                     "Trying to Create embeddings from raw documents"
#                 )
#             )
#             local_path = f"{settings.LOCAL_DATA_PATH}/{settings.RAW_DOC_PATH}"
#             file_name = settings.RAW_DOC_PATH
#             vec_handler = get_vec_store_handler()
#             vec_handler.load_vecstore(obj_path=file_name, path=local_path)
#             vec_handler.update_vecstore()
#         except Exception as e:
#             msg = "Couldn't Create Vector Store at all!"
#             logger.exception(msg)
#             raise Exception(f"{msg} Error {str(e)}")

#     yield
