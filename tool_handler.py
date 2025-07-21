# tool_handler.py
import json


def get_sep(raw):
    return next((s for s in ['[TOOL_CALLS]', '###', '<-->', '```'] if s in raw), '')


class ToolHandler:
    def __init__(self, memory, fetcher):
        self.memory = memory
        self.fetcher = fetcher

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
                        print(user_id, obj["task"])  # optional: Rückgabe
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
                            elif action.get("type") == "crypto_price":
                                results = self.fetcher.get_crypto_price(action.get("args"))
                                template = self.memory.get_tpl_message("web_search", {
                                    "query": action.get("args", {}).get("ids"),
                                    "results": results,
                                })
                                next_prompt = template
            except Exception as e:
                print(f"[ToolHandler] JSON parsing failed: {e}")
        return answer.strip(), next_prompt
