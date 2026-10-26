"""Print a portable command from a recorded configuration. Never executes it."""
import argparse
import json
import os
import re
import shlex
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('config', type=Path)
args = parser.parse_args()
command = json.loads(args.config.read_text())
required = set(re.findall(r'\{([A-Z_]+)\}', '\n'.join(command)))
missing = required - os.environ.keys()
if missing:
    parser.error('Set these environment variables: ' + ', '.join(sorted(missing)))
for index, token in enumerate(command):
    command[index] = re.sub(r'\{([A-Z_]+)\}', lambda match: os.environ[match[1]], token)
print(shlex.join(command))
