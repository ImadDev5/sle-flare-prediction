import os
import re

def clean_ai_comments(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove AI-generated looking comments
    ai_patterns = [
        r'# Author:.*\n',
        r'# Date:.*\n',
        r'"""[\s\S]*?Author:[\s\S]*?"""',
        r'"""[\s\S]*?Date:[\s\S]*?"""',
        r'# This script.*\n',
        r'# Generate.*\n',
        r'# Create.*\n',
        r'# Setup.*\n',
        r'#!/usr/bin/env python3\n',
        r'"""[\s\S]*?🎯[\s\S]*?"""',
        r'"""[\s\S]*?===+[\s\S]*?"""',
    ]

    for pattern in ai_patterns:
        content = re.sub(pattern, '', content, flags=re.MULTILINE)

    # Clean up excessive newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.strip()

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

# Clean all Python files
python_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.py') and not file.startswith('cleanup_'):
            python_files.append(os.path.join(root, file))

for file_path in python_files:
    try:
        clean_ai_comments(file_path)
        print(f"Cleaned: {file_path}")
    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")

print("All Python files cleaned!")