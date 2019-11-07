import sys
import os
from pathlib import Path

TASK_TEMPLATE = """\
{{
    "version": "2.0.0",
    "args": [],
    "echoCommand": false,
    "tasks": [
        {{
            "group": "test",
            "label": "strict",
            "type": "shell",
            "command": "{tester:s}",
            "args": [
                "--mpl",
                "--pep8"
            ]
        }},
        {{
            "group": "test",
            "label": "quick",
            "type": "shell",
            "command": "{tester:s}",
            "args": [
                "--pep8"
            ]
        }},
        {{
            "label": "docs",
            "type": "shell",
            "options": {{
                "cwd": "${{workspaceRoot}}/docs"
            }},
            "command": "make.bat",
            "args": ["html"]
        }}
    ]
}}
"""

SETTINGS_TEMPLATE = """\
// Place your settings in this file to overwrite default and user settings.
{{
    "python.pythonPath": "{pyexec:s}",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "--mpl",
        "--cov",
    ]
}}
"""


if __name__ == "__main__":
    configdir = Path(".vscode")
    configdir.mkdir(exist_ok=True, parents=True)
    taskspath = configdir / "tasks.json"
    settingspath = configdir / "settings.json"

    if len(sys.argv) < 2:
        name = Path.cwd().name
    else:
        name = sys.argv[1]

    python = Path(sys.executable)
    tester = python.parent / 'pytest'

    with taskspath.open("w") as tasksfile:
        tasksfile.write(TASK_TEMPLATE.format(tester=str(tester), modulename=name))

    with settingspath.open("w") as settingsfile:
        settingsfile.write(SETTINGS_TEMPLATE.format(pyexec=str(python)))
