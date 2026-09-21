import os
import json
import re

# -------------------------------------------------
# Base Conversations Directory
# -------------------------------------------------

CHAT_DIR = "conversations"

os.makedirs(CHAT_DIR, exist_ok=True)


# -------------------------------------------------
# Helper: Get User Chat Directory
# -------------------------------------------------

def get_user_chat_dir(username):
    """
    Creates and returns a separate chat directory
    for each logged-in user.
    """

    # Remove unsafe characters from username
    safe_username = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        str(username)
    )

    user_dir = os.path.join(
        CHAT_DIR,
        safe_username
    )

    os.makedirs(
        user_dir,
        exist_ok=True
    )

    return user_dir


# -------------------------------------------------
# Helper: Make Safe Chat Filename
# -------------------------------------------------

def sanitize_chat_name(chat_name):
    """
    Converts an AI-generated chat title into a safe
    filename.
    """

    chat_name = str(chat_name).strip()

    # Remove characters that are unsafe in filenames
    chat_name = re.sub(
        r'[<>:"/\\|?*]',
        '',
        chat_name
    )

    # Replace multiple spaces with one
    chat_name = re.sub(
        r"\s+",
        " ",
        chat_name
    )

    # Limit filename length
    chat_name = chat_name[:80]

    # Fallback
    if not chat_name:
        chat_name = "New Chat"

    return chat_name


# -------------------------------------------------
# Helper: Make Objects JSON Serializable
# -------------------------------------------------

def make_json_serializable(obj):
    """
    Converts non-JSON-serializable objects into
    JSON-compatible Python objects.
    """

    # LangChain Document
    if hasattr(obj, "page_content") and hasattr(obj, "metadata"):

        return {
            "page_content": str(obj.page_content),
            "metadata": make_json_serializable(obj.metadata)
        }

    # Dictionary
    if isinstance(obj, dict):

        return {
            str(key): make_json_serializable(value)
            for key, value in obj.items()
        }

    # List
    if isinstance(obj, list):

        return [
            make_json_serializable(item)
            for item in obj
        ]

    # Tuple
    if isinstance(obj, tuple):

        return [
            make_json_serializable(item)
            for item in obj
        ]

    # Set
    if isinstance(obj, set):

        return [
            make_json_serializable(item)
            for item in obj
        ]

    # Basic JSON types
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Fallback
    return str(obj)


# -------------------------------------------------
# Save Chat
# -------------------------------------------------

def save_chat(username, chat_name, messages):
    """
    Saves a user's chat history as JSON.

    Automatically converts objects such as LangChain
    Document objects into JSON-compatible data.
    """

    user_dir = get_user_chat_dir(username)

    chat_name = sanitize_chat_name(chat_name)

    path = os.path.join(
        user_dir,
        f"{chat_name}.json"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2, default=make_json_serializable)


# -------------------------------------------------
# Load Chat
# -------------------------------------------------

def load_chat(username, chat_name):
    """
    Loads a previously saved chat.

    Returns an empty list if the chat does not exist.
    """

    user_dir = get_user_chat_dir(username)

    chat_name = sanitize_chat_name(chat_name)

    path = os.path.join(
        user_dir,
        f"{chat_name}.json"
    )

    if os.path.exists(path):

        try:

            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                return json.load(f)

        except (json.JSONDecodeError, OSError):

            return []

    return []


# -------------------------------------------------
# List Chats
# -------------------------------------------------

def list_chats(username):
    """
    Returns all saved chat names for the user.
    """

    user_dir = get_user_chat_dir(username)

    chats = []

    for file in os.listdir(user_dir):

        if file.endswith(".json"):

            chats.append(
                file[:-5]
            )

    # Alphabetical order
    chats.sort()

    return chats

# -------------------------------------------------
# Delete Chat
# -------------------------------------------------

def delete_chat(username, chat_name):
    """
    Deletes a saved chat for the logged-in user.
    Returns True if deleted successfully,
    otherwise False.
    """

    user_dir = get_user_chat_dir(username)
    chat_name = sanitize_chat_name(chat_name)

    path = os.path.join(
        user_dir,
        f"{chat_name}.json"
    )

    if os.path.exists(path):
        try:
            os.remove(path)
            return True
        except OSError:
            return False

    return False

# -------------------------------------------------
# Generate Default Chat Name
# -------------------------------------------------
def generate_chat_name(question=None):
    """
    Generates a safe chat name.

    If a question is provided, the first few words
    are used as the chat title.
    """

    if question:

        words = str(question).strip().split()

        chat_name = " ".join(words[:6])

        if len(words) > 6:
            chat_name += "..."

        return sanitize_chat_name(chat_name)

    return "New Chat"