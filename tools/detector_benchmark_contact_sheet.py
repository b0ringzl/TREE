"""Analytical contact sheets of actual saved detector outputs, not synthetic detections."""
from pathlib import Path
from PIL import Image, ImageDraw
from tools.benchmark_tree_detectors import OUT, MODELS


def main():
    target = OUT / 'case_sheets'
    target.mkdir(exist_ok=True)
    frames = sorted((OUT / 'overlays/project_baseline/val').glob('*.jpg'))
    for frame in frames:
        sheet = Image.new('RGB', (1800, 1140), '#f5f5f5')
        draw = ImageDraw.Draw(sheet)
        for index, key in enumerate(MODELS):
            image_path = OUT / 'overlays' / key / 'val' / frame.name
            if not image_path.exists():
                continue
            with Image.open(image_path) as original:
                # Preserve full panorama for context; zoom should be inspected separately.
                image = original.convert('RGB')
                image.thumbnail((590, 340))
            x, y = (index % 3)*600, (index//3)*380
            draw.text((x+6, y+6), key, fill='black')
            sheet.paste(image, (x+5, y+30))
        sheet.save(target / frame.name, quality=93)
    print(target)


if __name__ == '__main__':
    main()
