from typing import Optional

from knowledge_base.runtime import datastore
from knowledge_base.utils import generate_api_key, hash_password


async def register_user(email: str, password: str):
    existing = await datastore.get_user_by_email(email)
    if existing:
        raise ValueError("User already exists")
    api_key = generate_api_key()
    password_hash = hash_password(password)
    user_id = await datastore.create_user(email, password_hash, api_key)
    return {"id": user_id, "email": email, "api_key": api_key}


async def login_user(email: str, password: str):
    user = await datastore.get_user_by_email(email)
    if not user:
        raise ValueError("Invalid credentials")
    if user.get("password_hash") != hash_password(password):
        raise ValueError("Invalid credentials")
    return {"id": user.get("id"), "email": user.get("email"), "api_key": user.get("api_key")}


async def get_user_by_api_key(api_key: str):
    return await datastore.get_user_by_api_key(api_key)


async def get_user_by_id(user_id: str):
    return await datastore.get_user_by_id(user_id)


def is_authorized(
    assistant: dict,
    email: Optional[str],
    password: Optional[str],
    api_key: Optional[str],
    owner_override: bool,
) -> bool:
    if not assistant.get("secure_enabled"):
        return True
    if owner_override:
        return True
    if api_key and api_key == assistant.get("api_key"):
        return True
    creds = assistant.get("credentials") or []
    for cred in creds:
        if cred.get("email") == email and cred.get("password") == password:
            return True
    return False
