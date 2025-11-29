<CLAUDE_PROJECT_DIR> is the current project root. 

Set env param CLAUDE_PROJECT_DIR to <CLAUDE_PROJECT_DIR> when not existed.

Check if ${CLAUDE_PROJECT_DIR}/.claude/skill-rules.json exists at project root and create it if it doesn't.

First, check if the file exists:

```bash
if [ ! -f "${CLAUDE_PROJECT_DIR}/.claude/skill-rules.json" ]; then
    echo "File ${CLAUDE_PROJECT_DIR}/.claude/skill-rules.json does not exist. Creating it..."
    touch ${CLAUDE_PROJECT_DIR}/.claude/skill-rules.json
    echo "Created ${CLAUDE_PROJECT_DIR}/.claude/skill-rules.json"
else
    echo "File ${CLAUDE_PROJECT_DIR}/.claude/skill-rules.json already exists"
fi
```

Then, update skill-rules.json by collect all skill's skill-rules.json, plugin-root is

```bash
echo ${CLAUDE_PLUGIN_ROOT}
echo ${CLAUDE_PROJECT_DIR}
${CLAUDE_PLUGIN_ROOT}/meta/force-trigger/scripts/update_skill_rules.py --plugin-root ${CLAUDE_PLUGIN_ROOT} --output ${CLAUDE_PROJECT_DIR}/.claude/skill-rules.json
```