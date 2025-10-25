import os
from ast import literal_eval


def makedirs_with_log(path):
    """Creates a directory with a log message.
    Args:
        path (str): The path to the directory to create.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        print(f'Directory {path} could not be created, reason: {error}')


def get_files_from_folder(folder_path, extensions=None, name_filter=None):
    """Gets a list of files from a folder.
    Args:
        folder_path (str): The path to the folder.
        extensions (list, optional): A list of file extensions to include. Defaults to None.
        name_filter (str, optional): A string to filter filenames by. Defaults to None.
    Returns:
        list: A list of filenames.
    """
    if not os.path.isdir(folder_path):
        raise ValueError("Folder path is not a valid directory.")

    filenames = []

    for root, _, files in os.walk(folder_path, topdown=False):
        relative_path = os.path.relpath(root, folder_path)
        if relative_path == ".":
            relative_path = ""
        for filename in sorted(files, key=lambda s: s.casefold()):
            _, file_extension = os.path.splitext(filename)
            if (extensions is None or file_extension.lower() in extensions) and (name_filter is None or name_filter in _):
                path = os.path.join(relative_path, filename)
                filenames.append(path)

    return filenames


def try_eval_env_var(value: str, expected_type=None):
    """Tries to evaluate an environment variable.
    Args:
        value (str): The value of the environment variable.
        expected_type (type, optional): The expected type of the environment variable. Defaults to None.
    Returns:
        The evaluated environment variable, or the original value if evaluation fails.
    """
    try:
        value_eval = value
        if expected_type is bool:
            value_eval = value.title()
        value_eval = literal_eval(value_eval)
        if expected_type is not None and not isinstance(value_eval, expected_type):
            return value
        return value_eval
    except:
        return value
