"""
Component 2: Target Registry.

Customers register their LLM applications here. We store the metadata
in the database and the API key in the vault (separately, so the key
is never in the same place as the endpoint URL).
"""

from . import storage
from .types import Target, new_id


def register_target(
    name: str,
    endpoint_url: str,
    api_key: str,
    system_type: str,
    purpose: str,
) -> Target:
    """Register a new LLM target. Returns the Target (without the api_key)."""
    target_id = new_id("tgt")
    auth_ref = f"vault://targets/{target_id}/api_key"

    storage.vault_put(auth_ref, api_key)

    target = Target(
        id=target_id,
        name=name,
        endpoint_url=endpoint_url,
        system_type=system_type,
        purpose=purpose,
        auth_ref=auth_ref,
    )
    storage.save_target(target)
    return target


def get_target(target_id: str) -> Target:
    """Fetch a target. The api_key is NOT included — workers fetch it separately."""
    t = storage.get_target(target_id)
    if not t:
        raise ValueError(f"Target {target_id} not found")
    return t


def get_api_key(target: Target) -> str:
    """Workers call this just-in-time when they need to invoke the target.
    In production, this would mint a short-lived scoped token instead."""
    return storage.vault_get(target.auth_ref)
