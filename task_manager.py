# task_manager.py
from datetime import datetime
from typing import Optional
import automations

class Task:
    def __init__(self,
                 title: str,
                 prompt: str,
                 jawbone_id: Optional[str] = None):
        self.title = title
        self.prompt = prompt
        self.jawbone_id = jawbone_id

class TaskManager:
    def __init__(self):
        # Merkt sich alle aktiven Tasks: title -> Task
        self.tasks: dict[str, Task] = {}

    def create_reminder(self,
                        title: str,
                        prompt: str,
                        when: datetime) -> None:
        """
        Erstellt eine einmalige Erinnerung.
        :param title:  Kurz und imperativ, z.B. "Check email"
        :param prompt: "Tell me to check my email."
        :param when:   datetime der Erinnerung (UTC)
        """
        # Build VEVENT
        dt = when.strftime("%Y%m%dT%H%M%SZ")
        schedule = (
            "BEGIN:VEVENT\n"
            f"DTSTART:{dt}\n"
            "END:VEVENT"
        )
        resp = automations.create(
            title=title,
            prompt=prompt,
            schedule=schedule
        )
        jid = resp["jawbone_id"]
        self.tasks[title] = Task(title, prompt, jawbone_id=jid)
        print(f"Reminder '{title}' created, jawbone_id={jid}")

    def create_periodic(self,
                        title: str,
                        prompt: str,
                        freq: str,
                        hour: int,
                        minute: int = 0) -> None:
        """
        Erstellt einen wiederkehrenden Task.
        :param freq: "DAILY" | "WEEKLY" | "MONTHLY"
        :param hour: Stunde im 24h‐Format
        :param minute: Minute
        """
        rule = f"RRULE:FREQ={freq};BYHOUR={hour};BYMINUTE={minute};BYSECOND=0"
        schedule = f"BEGIN:VEVENT\n{rule}\nEND:VEVENT"
        resp = automations.create(
            title=title,
            prompt=prompt,
            schedule=schedule
        )
        jid = resp["jawbone_id"]
        self.tasks[title] = Task(title, prompt, jawbone_id=jid)
        print(f"{freq}-Task '{title}' created, jawbone_id={jid}")

    def update_task_time(self,
                         title: str,
                         new_when: datetime) -> None:
        """
        Verschiebt einen bestehenden Task auf ein neues Datum/Uhrzeit.
        """
        task = self.tasks.get(title)
        if not task:
            raise KeyError(f"No task with title '{title}'")
        dt = new_when.strftime("%Y%m%dT%H%M%SZ")
        schedule = (
            "BEGIN:VEVENT\n"
            f"DTSTART:{dt}\n"
            "END:VEVENT"
        )
        automations.update(
            jawbone_id=task.jawbone_id,
            schedule=schedule
        )
        print(f"Task '{title}' moved to {new_when.isoformat()}")

    def cancel(self, title: str) -> None:
        """
        Deaktiviert (löst) einen bestehenden Task.
        """
        task = self.tasks.get(title)
        if not task:
            raise KeyError(f"No task with title '{title}'")
        automations.update(
            jawbone_id=task.jawbone_id,
            is_enabled=False
        )
        print(f"Task '{title}' cancelled")
