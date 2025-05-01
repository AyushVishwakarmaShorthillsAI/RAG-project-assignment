import re

file_path = 'test_cases_2.txt'

with open(file_path, 'r') as f:
    content = f.read()

# This counts opening braces `{` that likely start objects in an array
count = len(re.findall(r'\{\s*"Question"\s*:', content))

print(f"Number of objects: {count}")
