import re
import sys
from pathlib import Path

import yaml

RULES_TEMPLATE = """
{
    "{skill_name}": {
        "type": "",
        "enforcement": "",
        "priority": "",
        "promptTriggers": {
            "keywords": ["", ""],
            "intentPatterns": [""]
    }
}
"""


def main():
    # check if SKILL.md exists in the skill path
    skill_path = Path(sys.argv[1])
    skill_md = skill_path / "SKILL.md"
    if not skill_md.exists():
        return False, f"SKILL.md does not exist in the skill path: {skill_path}"

    # Read and validate frontmatter
    content = skill_md.read_text()
    if not content.startswith("---"):
        return False, "No YAML frontmatter found"

    # Extract frontmatter
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return False, "Invalid frontmatter format"

    frontmatter_text = match.group(1)

    # Parse YAML frontmatter
    try:
        frontmatter = yaml.safe_load(frontmatter_text)
        if not isinstance(frontmatter, dict):
            return False, "Frontmatter must be a YAML dictionary"
    except yaml.YAMLError as e:
        return False, f"Invalid YAML in frontmatter: {e}"

    # Check required fields
    if "name" not in frontmatter:
        return False, "Missing 'name' in frontmatter"
    if "description" not in frontmatter:
        return False, "Missing 'description' in frontmatter"

    # Extract name for validation
    name = frontmatter.get("name", "")
    if not isinstance(name, str):
        return False, f"Name must be a string, got {type(name).__name__}"
    name = name.strip()

    # Extract and validate description
    description = frontmatter.get("description", "")
    if not isinstance(description, str):
        return False, f"Description must be a string, got {type(description).__name__}"
    description = description.strip()

    print(f"📌 Skill Name: {name}")
    print(f"📌 Skill Description: {description}")
    print(f"🚀 Initializing {skill_path}/skill-rules.json")

    skill_rules_path = skill_path / "skill-rules.json"
    skill_rules_content = RULES_TEMPLATE.format(skill_name=name)
    try:
        skill_rules_path.write_text(skill_rules_content)
        return True, f"Created skill rules file: {skill_rules_path}"
    except Exception as e:
        return False, f"Error creating skill rules file: {e}"


if __name__ == "__main__":
    result, message = main()
    if result:
        print(f"✅ {message}")
        sys.exit(0)
    else:
        print(f"❌ {message}")
        sys.exit(1)
