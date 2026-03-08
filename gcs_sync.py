"""
gcs_sync.py - Optional Google Cloud Storage sync for the SQLite database.

When ``GCS_BUCKET_NAME`` is set in the environment the local SQLite file is:

* **downloaded** at startup (if the remote blob already exists), so that state
  persists across re-deployments.
* **uploaded** after every ``DatabaseManager.save_*`` call, so that the latest
  state is always mirrored in the cloud.

Configuration (via environment variables or Streamlit secrets):

``GCS_BUCKET_NAME``
    Name of the GCS bucket to use (required to enable sync).

``GCS_DB_BLOB_NAME``
    Object name inside the bucket.  Defaults to ``pse_trading_bot.db``.

``GCS_CREDENTIALS_JSON``
    Optional.  A JSON string containing a Google service-account credentials
    object.  Use this when ``GOOGLE_APPLICATION_CREDENTIALS`` cannot point to a
    file (e.g. Streamlit Cloud, Heroku).  If omitted the SDK falls back to
    Application Default Credentials.
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENV_BUCKET = "GCS_BUCKET_NAME"
_ENV_BLOB = "GCS_DB_BLOB_NAME"
_DEFAULT_BLOB = "pse_trading_bot.db"
_ENV_CREDS_JSON = "GCS_CREDENTIALS_JSON"


def _get_bucket_name() -> Optional[str]:
    return os.environ.get(_ENV_BUCKET) or None


def _get_blob_name() -> str:
    return os.environ.get(_ENV_BLOB) or _DEFAULT_BLOB


def _build_client():
    """Return a google.cloud.storage.Client, handling JSON-string credentials."""
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "google-cloud-storage is required for GCS sync. "
            "Install it with: pip install google-cloud-storage"
        ) from exc

    creds_json = os.environ.get(_ENV_CREDS_JSON)
    if creds_json:
        info = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/devstorage.read_write"],
        )
        return storage.Client(credentials=credentials, project=info.get("project_id"))

    # Fall back to Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS)
    return storage.Client()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_gcs_enabled() -> bool:
    """Return True when GCS sync is configured (``GCS_BUCKET_NAME`` is set)."""
    return bool(_get_bucket_name())


def upload_db(db_path: str) -> None:
    """Upload the local SQLite file to GCS.

    A no-op when :func:`is_gcs_enabled` returns False.

    Args:
        db_path: Absolute path to the local ``.db`` file.
    """
    bucket_name = _get_bucket_name()
    if not bucket_name:
        return
    if not os.path.exists(db_path):
        logger.warning("upload_db: file not found, skipping: %s", db_path)
        return

    try:
        client = _build_client()
        blob = client.bucket(bucket_name).blob(_get_blob_name())
        blob.upload_from_filename(db_path)
        logger.info("DB uploaded to gs://%s/%s", bucket_name, _get_blob_name())
    except Exception as exc:  # pragma: no cover – network errors
        logger.error("GCS upload failed: %s", exc)


def download_db(db_path: str) -> bool:
    """Download the SQLite file from GCS to *db_path*.

    A no-op when :func:`is_gcs_enabled` returns False.

    Args:
        db_path: Absolute path where the downloaded file should be written.

    Returns:
        True if the file was downloaded successfully, False otherwise.
    """
    bucket_name = _get_bucket_name()
    if not bucket_name:
        return False

    try:
        client = _build_client()
        blob = client.bucket(bucket_name).blob(_get_blob_name())
        if not blob.exists():
            logger.info("No remote DB found at gs://%s/%s — starting fresh.", bucket_name, _get_blob_name())
            return False
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        blob.download_to_filename(db_path)
        logger.info("DB downloaded from gs://%s/%s", bucket_name, _get_blob_name())
        return True
    except Exception as exc:  # pragma: no cover – network errors
        logger.error("GCS download failed: %s", exc)
        return False
