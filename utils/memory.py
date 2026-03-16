chat_history = []


def add_to_memory(question, answer, max_history=5):
    global chat_history

    chat_history.append((question, answer))

    if len(chat_history) > max_history:
        chat_history.pop(0)


def get_history():
    return chat_history


def format_history():

    history_text = ""

    for q, a in chat_history:
        history_text += f"User: {q}\nAssistant: {a}\n"

    return history_text


def clear_memory():
    global chat_history
    chat_history = []