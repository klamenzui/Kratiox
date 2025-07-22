import uuid
import logging

def create(title: str, prompt: str, schedule: str) -> dict:
    logging.info("[AUTOMATIONS] Create: %s\nPrompt: %s\nSchedule:\n%s", title, prompt, schedule)
    return {"jawbone_id": str(uuid.uuid4())}

def update(jawbone_id: str, schedule: str = None, is_enabled: bool = True) -> None:
    logging.info("[AUTOMATIONS] Update: %s\nSchedule: %s\nEnabled: %s", jawbone_id, schedule, is_enabled)
