import json
import datetime

def log_session(question, retrieved_context, answer, log_path="session_logs.jsonl"):
    log_entry = {
        "datetime": datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "question": question,
        "retrieved_context": retrieved_context[:400],
        "answer": answer
    }
    with open(log_path, mode="a", encoding="utf-8") as logfile:
        logfile.write(json.dumps(log_entry, ensure_ascii=False) + "\n")