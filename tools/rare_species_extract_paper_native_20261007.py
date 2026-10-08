"""Lossless extraction of original PDF photograph pixels, no rendering/upscale."""
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/rare_species_fresh_validation_20261006_v1'

def main():
    results = []
    for pmcid, page_index, image_name in [('PMC7050300', 2, 'Im2.jpg'), ('PMC10935121', 3, 'Im4.jpg')]:
        dest = OUT / 'paper_originals' / (pmcid + '_cloud_v1')
        pdf = next(dest.glob('*.pdf'))
        reader = PdfReader(pdf)
        image = next(im for im in reader.pages[page_index].images if im.name == image_name)
        path = dest / 'figure1_native_pdf.jpg'
        if path.exists(): raise FileExistsError(path)
        path.write_bytes(image.data)
        results.append({'pmcid': pmcid, 'pdf_file': pdf.relative_to(OUT).as_posix(),
                        'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
                        'page_index': page_index, 'pdf_image_name': image_name,
                        'file': path.relative_to(OUT).as_posix(), 'dimensions': list(image.image.size),
                        'sha256': hashlib.sha256(image.data).hexdigest(), 'resampled': False})
    (OUT / 'paper_native_extraction.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(results)

if __name__ == '__main__': main()
