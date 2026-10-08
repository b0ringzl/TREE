import json

with open('pose_templates.json', 'r', encoding='utf-8') as f:
    templates = json.load(f)

for t in templates:
    name = t.get('name', '')
    if 'Person' in name:
        t['label'] = 'person'
    elif 'Hand' in name:
        t['label'] = 'hand'
    elif 'Face' in name:
        t['label'] = 'face'
    else:
        t['label'] = name.lower().replace(' ', '_')
    
    kpt_count = len(t.get('keypoints', []))
    t['kpt_shape'] = [kpt_count, 3]

# reorder keys so name, label, description, kpt_shape are first
ordered_templates = []
for t in templates:
    ordered = {
        "name": t.get("name"),
        "label": t.get("label"),
        "description": t.get("description", ""),
        "kpt_shape": t.get("kpt_shape"),
        "keypoints": t.get("keypoints", []),
        "connections": t.get("connections", [])
    }
    ordered_templates.append(ordered)

with open('pose_templates.json', 'w', encoding='utf-8') as f:
    json.dump(ordered_templates, f, ensure_ascii=False, indent=4)

print("Updated pose_templates.json")
