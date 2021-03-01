import os

def append_line(file_path, line):
    file = open(file_path, 'a')
    file.write(line + "\n")
    file.close()

def clear_file(file_path):
    open(file_path, 'w').close()

def get_lines(file_path):
    file = open(file_path, 'r')
    lines = file.read().splitlines()
    return lines

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)