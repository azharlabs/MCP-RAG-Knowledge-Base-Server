import shutil

from knowledge_base.runtime import LOCAL_STORAGE_PATH, STORAGE_TYPE, bucket
from extractanything import ExtractAnything

extract_extractor = ExtractAnything()


def store_file(temp_path: str, assistant_id: str, filename: str) -> str:
    if STORAGE_TYPE == "gcs" and bucket:
        blob = bucket.blob(f"{assistant_id}/{filename}")
        blob.upload_from_filename(temp_path)
        return blob.public_url

    if not LOCAL_STORAGE_PATH:
        raise RuntimeError("Local storage path not configured")
    dest_dir = LOCAL_STORAGE_PATH / assistant_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    shutil.copy(temp_path, dest_path)
    return str(dest_path)


def extract_text_with_extractanything(file_path: str) -> str:
    return extract_extractor.extract_text(file_path)
