Check if .claude/skill-rules.json exists at project root and create it if it doesn't.

First, check if the file exists:

```bash
if [ ! -f ".claude/skill-rules.json" ]; then
    echo "File .claude/skill-rules.json does not exist. Creating it..."
    touch .claude/skill-rules.json
    echo "Created .claude/skill-rules.json"
else
    echo "File .claude/skill-rules.json already exists"
fi
```

Then, update skill-rules.json by collect all skill's skill-rules.json:

```bash
echo ${CLAUDE_PLUGIN_ROOT}
echo ${CLAUDE_PROJECT_DIR}
${CLAUDE_PLUGIN_ROOT}/meta/force-trigger/scripts/update_skill_rules.py --plugin-root ${CLAUDE_PLUGIN_ROOT}
```