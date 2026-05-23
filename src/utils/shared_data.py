import threading

class DeepThreadSafeDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Reentrant lock to avoid nested lock deadlock
        self.lock = threading.RLock()

    def __getitem__(self, key):
        with self.lock:
            value = super().__getitem__(key)
            # Wrap native dict/list as proxy objects (avoid circular proxy)
            if isinstance(value, dict) and not isinstance(value, DeepThreadSafeDict):
                return _DictProxy(self, key, value)
            elif isinstance(value, list) and not isinstance(value, _ListProxy):
                return _ListProxy(self, key, value)
            return value

    def __setitem__(self, key, value):
        with self.lock:
            # Auto-convert native dict to DeepThreadSafeDict for thread safety
            if isinstance(value, dict) and not isinstance(value, DeepThreadSafeDict):
                value = DeepThreadSafeDict(value)
            # Extract raw list from ListProxy before assignment
            elif isinstance(value, _ListProxy):
                value = value.target_list.copy()
            # Extract raw dict from DictProxy before assignment
            elif isinstance(value, _DictProxy):
                value = DeepThreadSafeDict(value.target_dict)
            super().__setitem__(key, value)

    def update(self, other_dict):
        """
        Thread-safe deep update method
        :param other_dict: Dictionary containing update data
        :return: None
        """
        with self.lock:
            self._deep_update(self, other_dict)

    def _deep_update(self, target, source):
        """
        Recursively update dictionary with deep thread safety
        Fixed: Correctly handle proxy objects and avoid wrong type conversion
        :param target: Target dictionary to be updated
        :param source: Source dictionary providing update data
        :return: None
        """
        for key, value in source.items():
            # Get raw value (unproxy) from target first
            target_value = self._get_raw_value(target.get(key))
            
            # Case 1: Both source and target are dictionaries (recursive update)
            if isinstance(value, dict) and isinstance(target_value, dict):
                # Ensure target is DeepThreadSafeDict
                if not isinstance(target[key], DeepThreadSafeDict):
                    target[key] = DeepThreadSafeDict(target_value)
                self._deep_update(target[key], value)
            
            # Case 2: Both source and target are lists (extend instead of overwrite)
            elif isinstance(value, list) and isinstance(target_value, list):
                target[key].extend(value)
            
            # Case 3: Overwrite for other types (including new keys)
            else:
                # Auto-wrap dict to DeepThreadSafeDict
                if isinstance(value, dict) and not isinstance(value, DeepThreadSafeDict):
                    target[key] = DeepThreadSafeDict(value)
                # Copy list to avoid reference sharing
                elif isinstance(value, list):
                    target[key] = value.copy()
                # Direct assignment for basic types
                else:
                    target[key] = value

    def _get_raw_value(self, value):
        """
        Extract raw data from proxy objects (internal helper)
        :param value: May be native type/dict/list or proxy object
        :return: Raw underlying value
        """
        if isinstance(value, _DictProxy):
            return value.target_dict
        elif isinstance(value, _ListProxy):
            return value.target_list
        elif isinstance(value, DeepThreadSafeDict):
            # Convert DeepThreadSafeDict to native dict for type check
            return dict(value)
        else:
            return value

# Dictionary proxy class (fixed lock and parent reference)
class _DictProxy:
    def __init__(self, parent_dict, key, target_dict):
        self.parent_dict = parent_dict  # Reference to top-level DeepThreadSafeDict
        self.key = key
        self.target_dict = target_dict  # Raw native dict

    def __getitem__(self, key):
        with self.parent_dict.lock:
            value = self.target_dict[key]
            if isinstance(value, dict):
                return _DictProxy(self.parent_dict, f"{self.key}.{key}", value)
            elif isinstance(value, list):
                return _ListProxy(self.parent_dict, self.key, value)
            return value

    def __setitem__(self, key, value):
        with self.parent_dict.lock:
            self.target_dict[key] = value
            self.parent_dict[self.key] = DeepThreadSafeDict(self.target_dict)

    def update(self, other_dict):
        with self.parent_dict.lock:
            self.parent_dict._deep_update(self.target_dict, other_dict)
            self.parent_dict[self.key] = DeepThreadSafeDict(self.target_dict)

# List proxy class (fixed lock and parent reference)
class _ListProxy:
    def __init__(self, parent_dict, key, target_list):
        self.parent_dict = parent_dict  # Reference to top-level DeepThreadSafeDict
        self.key = key
        self.target_list = target_list  # Raw native list

    def append(self, item):
        with self.parent_dict.lock:
            self.target_list.append(item)
            self.parent_dict[self.key] = self.target_list.copy()

    def extend(self, items):
        with self.parent_dict.lock:
            self.target_list.extend(items)
            self.parent_dict[self.key] = self.target_list.copy()

    def __setitem__(self, index, value):
        with self.parent_dict.lock:
            self.target_list[index] = value
            self.parent_dict[self.key] = self.target_list.copy()

    def insert(self, index, item):
        with self.parent_dict.lock:
            self.target_list.insert(index, item)
            self.parent_dict[self.key] = self.target_list.copy()


# Global result variable, used to store review results, etc.
# Instantiate thread-safe dictionary for import by other scripts
result = DeepThreadSafeDict({
    "reviews": {},
    "meta_review": {
        "content": None,
        "score": None,
        "final_decision": None
    }
})

paper = DeepThreadSafeDict({
    "id": None,
    "title": None, 
    "abstract": None,
    "content": None, # main text + image summary + appendix summary
    "main_text": None, # main text, without reference/appendix
    "reference": None,
    "appendix": None, # appendix summary
    "image": None, # image summary
    "images": {} # image list
})

# {
#     "reviews": {
#         "reviewer_1": {
#             "initial_review": {
#                 "content": "...",
#                 "score": 8,
#             },
#             "initial_malice_analysis": {
#                 "content": "...",
#                 "malice_types": ["conflict_of_interest"]
#             },
#             "rebuttal": {
#                 "content": "...",
#             },
#             "updated_review": [
#                 {
#                     "type": "review",
#                     "content": "...",
#                     "score": 9,
#                 },
#                 {
#                     "type": "malice_analysis",
#                     "content": "...",
#                     "malice_types": ["conflict_of_interest"]
#                 }
#             ]
#         },
#         ...
#     },
#     "meta_review": {
#         "content": "...",
#         "score": 9,
#         "final_decision": "accept"
#     }
# }



# shared_data.py
# import threading

# class ThreadSafeDict(dict):
#     def __init__(self):
#         super().__init__()
#         self.lock = threading.Lock()

#     def __setitem__(self, key, value):
#         """Override assignment method to automatically acquire/release lock"""
#         with self.lock:
#             super().__setitem__(key, value)

#     def __getitem__(self, key):
#         """Override value retrieval method to automatically acquire/release lock"""
#         with self.lock:
#             return super().__getitem__(key)

#     # If you need other methods (such as update, del, etc.), you can override them similarly
#     def update(self, *args, **kwargs):
#         with self.lock:
#             super().update(*args, **kwargs)