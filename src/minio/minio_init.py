from typing import Any, Callable, Dict, Optional, Tuple, List
import logging
from minio import Minio
from minio.error import S3Error

class MinIO():
    def __init__(
        self,
        minio_config: Optional[Dict[str, Any]] = None,
    ):
        self.logger = self._setup_logger()
        self.minio_config = minio_config or self._default_minio_config()
        self.client = self._setup_minio_client()
        self._ensure_bucket_exists()

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup logger for MinIO client."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @staticmethod
    def _default_minio_config() -> Dict[str, Any]:
        """Default MinIO configuration."""
        return {
            'endpoint': 'minio:9000',
            'access_key': 'admin',
            'secret_key': 'admin123',
            'secure': False,
            'bucket_name': 'ml-models'
        }
    
    def _setup_minio_client(self) -> Minio:
        """Initialize MinIO client with error handling."""
        try:
            client = Minio(
                self.minio_config['endpoint'],
                access_key=self.minio_config['access_key'],
                secret_key=self.minio_config['secret_key'],
                secure=self.minio_config['secure']
            )
            self.logger.info("MinIO client initialized successfully")
            return client
        except Exception as e:
            self.logger.error(f"Failed to initialize MinIO client: {str(e)}")
            raise RuntimeError(f"MinIO initialization failed: {str(e)}") from e
        
    def _ensure_bucket_exists(self) -> None:
        """Ensure MinIO bucket exists, create if not."""
        bucket_name = self.minio_config['bucket_name']
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                self.logger.info(f"Created MinIO bucket: {bucket_name}")
            else:
                self.logger.info(f"Using existing MinIO bucket: {bucket_name}")
        except S3Error as e:
            self.logger.error(f"Failed to setup bucket: {str(e)}")
            raise