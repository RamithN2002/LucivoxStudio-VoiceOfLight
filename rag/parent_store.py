parent_store = {}


def add_parent(parent_id, text):
    parent_store[parent_id] = text


def get_parent(parent_id):
    return parent_store.get(parent_id)


def get_multiple_parents(parent_ids):
    return [parent_store[p] for p in parent_ids if p in parent_store]