import os

path = "data/"

folders = [
    name for name in os.listdir(path)
    if os.path.isdir(os.path.join(path, name))
]

print(folders)
