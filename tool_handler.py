# tool_handler.py mit periodic task support
import json
import logging
from task_manager import TaskManager

logging.basicConfig(level=logging.INFO)

def get_sep(msg: str) -> str:
    for sep in ["[TOOL_CALLS]", "<-->", "###", "```"]:
        if sep in msg:
            return sep
    return ""

class ToolHandler:
    def __init__(self, memory, fetcher):
        self.memory = memory
        self.fetcher = fetcher
        self.task_manager = TaskManager()

    def process_llm_response(self, raw: str, chat_id: str, user_id: str, searched: bool = False):
        sep = get_sep(raw)
        answer = raw
        next_prompt = None

        if sep:
            try:
                answer, payload_raw = raw.split(sep, 1)
                start, end = payload_raw.find("{"), payload_raw.rfind("}")
                if start != -1 and end != -1:
                    js = payload_raw[start:end + 1]
                    obj = json.loads(js)

                    if obj.get("memory"):
                        self.memory.store_facts(obj["memory"])
                    if obj.get("settings"):
                        self.memory.store_settings(obj["settings"])
                    if obj.get("task"):
                        self._handle_task(obj["task"], user_id)
                    if obj.get("action"):
                        action = obj["action"]
                        if action.get("name") == "search" and not searched:
                            if action.get("type") == "text":
                                query = action.get("args", {}).get("query")
                                results = self.fetcher.web_search(action.get("args"))
                                template = self.memory.get_tpl_message("web_search", {
                                    "query": query,
                                    "results": results
                                })
                                next_prompt = template
            except Exception as e:
                logging.warning("[ToolHandler] JSON parsing failed: %s", e)

        return answer.strip(), next_prompt

    def _handle_task(self, task: dict, user_id: str):
        try:
            title = task.get("title")
            prompt = task.get("details") or task.get("prompt") or title
            task_type = task.get("type", "reminder")

            if task_type == "reminder":
                iso_time = task.get("time")
                if not iso_time:
                    logging.warning("No 'time' field in reminder task: %s", task)
                    return
                from datetime import datetime
                when = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
                self.task_manager.create_reminder(title, prompt, when)
                logging.info("Reminder scheduled for %s: %s", user_id, title)

            elif task_type == "periodic":
                freq = task.get("freq", "DAILY").upper()
                hour = int(task.get("hour", 8))
                minute = int(task.get("minute", 0))
                self.task_manager.create_periodic(title, prompt, freq, hour, minute)
                logging.info("Periodic task for %s: %s (%s @ %02d:%02d)", user_id, title, freq, hour, minute)

            else:
                logging.warning("Unknown task type: %s", task_type)
        except Exception as e:
            logging.error("Failed to schedule task: %s", e)