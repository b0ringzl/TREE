"""Create offline inspection sheets from already downloaded browser assets."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/conservation_browser_visual_qa_20260917_v1/red'
BUNDLES = [
    ('Herbarium', 'd87718c0-c25a-4ec4-b394-1dfb2b9a032b'),
    ('Pukeiti', '041ccc59-01f7-450d-b6ab-7001e08ce4b7'),
    ('Greening', 'cc5c150f-534a-438b-92b9-9c3f86800cb1'),
]

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    rows = []
    for source, bundle in BUNDLES:
        manifest = Path('C:/Users/250417~1/AppData/Local/Temp/browser-use/assets') / bundle / 'manifest.json'
        for asset in json.loads(manifest.read_text(encoding='utf-8'))['assets']:
            if asset['kind'] == 'image':
                rows.append(dict(asset, source=source, number=len(rows)+1))
    (OUT / 'image_index.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    for start in range(0, len(rows), 9):
        sheet = Image.new('RGB', (1500, 1500), 'white')
        draw = ImageDraw.Draw(sheet)
        for offset, row in enumerate(rows[start:start+9]):
            with Image.open(row['path']) as image:
                im = ImageOps.exif_transpose(image).convert('RGB')
                size = im.size
                im.thumbnail((486, 430))
            x, y = (offset % 3)*500, (offset // 3)*500
            sheet.paste(im, (x+(490-im.width)//2, y))
            draw.text((x+5,y+437), f"{row['number']:02d} | {row['source']} | {row['name']}", fill='black')
            draw.text((x+5,y+458), f"Native {size} | Rhodoleia championii | source label", fill='black')
        file = OUT / f'contact_{start//9+1}.jpg'
        sheet.save(file, quality=94)
        print(file)

if __name__ == '__main__':
    main()
