"""
Registry for LLM applications being red-teamed.

Target metadata (name, URL, system type) goes into the database.
The API key goes into the vault under a separate reference — so if the
database were ever exposed, it would not reveal any credentials.
"""

import logging
from . import storage
from .types import Target, new_id

logger = logging.getLogger(__name__)


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
    logger.debug("API key stored in vault at %s", auth_ref)

    target = Target(
        id=target_id,
        name=name,
        endpoint_url=endpoint_url,
        system_type=system_type,
        purpose=purpose,
        auth_ref=auth_ref,
    )
    storage.save_target(target)
    logger.info("Registered target %s | name='%s' system_type='%s' url='%s'",
                target.id, name, system_type, endpoint_url)
    return target


def get_target(target_id: str) -> Target:
    """Fetch a target. The api_key is NOT included — workers fetch it separately."""
    t = storage.get_target(target_id)
    if not t:
        logger.error("Target '%s' not found in registry", target_id)
        raise ValueError(f"Target {target_id} not found")
    logger.debug("Fetched target %s ('%s')", t.id, t.name)
    return t


def get_api_key(target: Target) -> str:
    """Fetch the target's API key from the vault.

    Workers call this just before invoking the target — not earlier.
    The key is held in memory only for the duration of the job.
    """
    key = storage.vault_get(target.auth_ref)
    logger.debug("Retrieved API key for target %s from %s", target.id, target.auth_ref)
    return key
