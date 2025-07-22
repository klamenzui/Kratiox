# task_manager.py
from datetime import datetime
from typing import Optional
import logging
import automations

logging.basicConfig(level=logging.INFO)

class Task:
    def __init__(self, title: str, prompt: str, jawbone_id: Optional[str] = None):
        self.title = title
        self.prompt = prompt
        self.jawbone_id = jawbone_id

class TaskManager:
    def __init__(self):
        self.tasks: dict[str, Task] = {}

    def create_reminder(self, title: str, prompt: str, when: datetime) -> None:
        dt = when.strftime("%Y%m%dT%H%M%SZ")
        schedule = (
            "BEGIN:VEVENT\n"
            f"DTSTART:{dt}\n"
            "END:VEVENT"
        )
        resp = automations.create(title=title, prompt=prompt, schedule=schedule)
        jid = resp["jawbone_id"]
        self.tasks[title] = Task(title, prompt, jawbone_id=jid)
        logging.info("Reminder '%s' created for %s", title, dt)

    def create_periodic(self, title: str, prompt: str, freq: str, hour: int, minute: int = 0) -> None:
        rule = f"RRULE:FREQ={freq};BYHOUR={hour};BYMINUTE={minute};BYSECOND=0"
        schedule = f"BEGIN:VEVENT\n{rule}\nEND:VEVENT"
        resp = automations.create(title=title, prompt=prompt, schedule=schedule)
        jid = resp["jawbone_id"]
        self.tasks[title] = Task(title, prompt, jawbone_id=jid)
        logging.info("Periodic task '%s' created (%s @ %02d:%02d)", title, freq, hour, minute)

    def update_task_time(self, title: str, new_when: datetime) -> None:
        task = self.tasks.get(title)
        if not task:
            raise KeyError(f"No task with title '{title}'")
        dt = new_when.strftime("%Y%m%dT%H%M%SZ")
        schedule = "BEGIN:VEVENT\nDTSTART:%s\nEND:VEVENT" % dt
        automations.update(jawbone_id=task.jawbone_id, schedule=schedule)
        logging.info("Task '%s' updated to %s", title, dt)

    def cancel(self, title: str) -> None:
        task = self.tasks.get(title)
        if not task:
            raise KeyError(f"No task with title '{title}'")
        automations.update(jawbone_id=task.jawbone_id, is_enabled=False)
        logging.info("Task '%s' cancelled", title)
