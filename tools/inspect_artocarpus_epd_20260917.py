"""Contact sheets for source PDF evidence only; not training data."""
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
out = root / 'external_datasets/artocarpus_whole_source_20260917_v1'
files = sorted((out / 'source_documents').glob('epd_femar2_page-*.jpg'))
sheet = Image.new('RGB', (1800, 3 * 670), 'white')
draw = ImageDraw.Draw(sheet)
for i, path in enumerate(files):
    with Image.open(path) as src:
        im = src.convert('RGB')
    im.thumbnail((440, 635))
    x, y = (i % 4) * 450, (i // 4) * 670
    sheet.paste(im, (x, y))
    draw.text((x+5, y+640), path.stem, fill='black')
target = out / 'contact_sheets/epd_femar2_evidence_pages.jpg'
if target.exists():
    raise FileExistsError(target)
sheet.save(target, quality=95)
