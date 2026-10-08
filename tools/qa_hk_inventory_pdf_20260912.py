from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
from pypdf import PdfReader
import json
ROOT=Path(__file__).resolve().parents[1];d=ROOT/'tmp/pdfs/hk_inventory_20260912'
paths=sorted(d.glob('page-*.png'));assert len(paths)==8
for i in range(0,8,2):
    canvas=Image.new('RGB',(1400,1020),'#d8e3da')
    for j,p in enumerate(paths[i:i+2]):
        with Image.open(p) as im:
            im.thumbnail((685,975));canvas.paste(im,(j*700+7,30))
        ImageDraw.Draw(canvas).text((j*700+14,10),'Page '+str(i+j+1),fill='black')
    canvas.save(d/f'contact_{i//2+1}.png')
reader=PdfReader(ROOT/'output/pdf/Hong_Kong_Data_Inventory_20260912.pdf')
texts=[p.extract_text() for p in reader.pages];alltext=''.join(texts)
checks=['72','95','20,691','131','66','293','147','04_summary.json','tree981','514.80']
assert all(s in alltext for s in checks)
(ROOT/'reports/hk_data_inventory_20260912_v1/pdf_qa.json').write_text(json.dumps(dict(pages=len(texts),critical_numbers_and_issues_present=True,page_text_lengths=list(map(len,texts)),rendered_pages=len(paths)),indent=2),encoding='utf-8')
