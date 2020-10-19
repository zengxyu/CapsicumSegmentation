import os


def check_and_create(path):
    """
    Create files in a loop
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
