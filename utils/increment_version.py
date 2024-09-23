import re


def increment_version(version):
    major, minor, patch = map(int, version.split('.'))
    patch += 1
    return f"{major}.{minor}.{patch}"


with open('version.txt', 'r+') as file:
    current_version = file.read().strip()
    new_version = increment_version(current_version)
    file.seek(0)
    file.write(new_version)
    file.truncate()

print(f"Version updated to {new_version}")
