import threading
import json  # New: For JSON serialization
from datetime import datetime  # New: For getting precise timestamps
from utils.logger import logger  # New: Import global message log path (from logger.py)

class MessageQueue:
    """Thread-safe message queue (supports multi-threaded parallel operations)"""
    def __init__(self):
        self.message_log_path = None
        self.messages = []
        self.lock = threading.Lock()  # Thread lock ensures read/write safety

    def set_message_log_path(self, path):
        self.message_log_path = path

    def send(self, sender, recipient, content, msg_type):
        with self.lock:  # Lock to ensure concurrent safety
            new_msg = {
                "sender": sender,
                "recipient": recipient,
                "content": content,
                "msg_type": msg_type,
                "timestamp": len(self.messages)
            }
            self.messages.append(new_msg)

            # New logic: Write message to JSONL file (append mode, one JSON per line)
            try:
                # Open file: a=append mode (does not overwrite historical messages), utf-8=supports Chinese
                with open(self.message_log_path, "a", encoding="utf-8") as f:
                    # Serialize JSON: ensure_ascii=False preserves Chinese characters to avoid garbled text
                    json.dump(new_msg, f, ensure_ascii=False)
                    f.write("\n")  # One message per line, compliant with JSONL format (facilitates subsequent parsing)
            except Exception as e:
                # Exception handling: Failed writes do not crash the program, only record error logs
                logger.error(f"[MessageQueue] Failed to write to message.jsonl: {str(e)}", exc_info=True)

            logger.debug(f"Message queue new entry: {sender} → {recipient} (Type: {msg_type})")

    # Core modification: Support recipient=None, which means no recipient filtering
    def receive(self, recipient=None, msg_type=None):
        with self.lock:
            # Step 1: Filter by recipient (no filtering if recipient is None)
            if recipient is not None:
                filtered = [msg for msg in self.messages if msg["recipient"] == recipient]
            else:
                filtered = self.messages  # No recipient restriction, keep all messages

            # Step 2: Filter by message type (optional)
            if msg_type:
                filtered = [msg for msg in filtered if msg["msg_type"] == msg_type]

            return filtered

    def load_messages(self, history_messages):
        required_fields = {"sender", "recipient", "content", "msg_type", "timestamp"}
        with self.lock:
            valid_messages = []
            for idx, msg in enumerate(history_messages):
                if not isinstance(msg, dict):
                    logger.warning(f"Skipping {idx+1} message: not a valid dict")
                    continue
                missing_fields = required_fields - msg.keys()
                if missing_fields:
                    logger.warning(f"Skipping {idx+1} message: missing fields {missing_fields}")
                    continue
                valid_messages.append(msg)
            
            # self.messages.clear()
            
            self.messages.extend(valid_messages)

            if self.message_log_path:
                try:
                    with open(self.message_log_path, "w", encoding="utf-8") as f:
                        for msg in valid_messages:
                            json.dump(msg, f, ensure_ascii=False)
                            f.write("\n")
                except Exception as e:
                    logger.error(f"[MessageQueue] History update failed: {str(e)}", exc_info=True)
            
            logger.info(f"Successfully loaded {len(valid_messages)}/{len(history_messages)} historical messages into the queue")

# Key: Create a global singleton queue, shared by all Agents
global_message_queue = MessageQueue()
