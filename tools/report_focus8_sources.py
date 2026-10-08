"""Read-only, self-contained official-photo contact gallery; no review writes."""
import argparse
import base64
import html
import io
from pathlib import Path
from PIL import Image
from distillation_core import read_json


def main(directory):
    cards=[]
    status_names={'accepted':'Previously accepted / 先前已接受', 'rejected':'Previously rejected / 先前已拒絕', 'pending':'Pending review / 待驗收'}
    for row in read_json(directory/'manifest.json'):
        with Image.open(row['file']) as im:
            im=im.convert('RGB');im.thumbnail((640,480))
            buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=90)
        uri='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')
        status=row['review_status']
        cards.append('<article class="'+html.escape(status)+'"><h2>'+html.escape(row['chinese'])+'</h2><p>'+html.escape(row['source_species'])+'</p><img src="'+uri+'"><p>'+html.escape(row['id'])+' · '+status_names.get(status,html.escape(status))+'</p><a target="_blank" href="'+html.escape(row['source_page'],quote=True)+'">Official source / 官方來源</a><p>© Hong Kong Herbarium, AFCD</p></article>')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>Official reference photos / 官方參考照片</title><style>body{font:16px system-ui;margin:28px;color:#153d32;background:#f1f5ef}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:18px}article{padding:18px;border:2px solid #c8d7c9;border-radius:12px;background:white}article.pending{border-color:#c29c4b}article.rejected{opacity:.65}img{width:100%;height:340px;object-fit:contain}button{padding:12px;margin:10px}</style><h1>Official reference photos / 官方參考照片</h1><p>45 source photos: 23 previously accepted, 4 previously rejected, 18 new pending photos. / 共45張：23張先前已接受、4張先前已拒絕、18張新增待驗收。</p><p>Read-only contact sheet. New photos have NOT been used for training. Original watermarked files are preserved. Quality acceptance does not grant reuse rights. / 唯讀對照表；新增照片尚未用於訓練，帶水印原圖完整保留；品質驗收不等於使用授權。</p><button onclick="document.querySelectorAll('article:not(.pending)').forEach(x=>x.hidden=!x.hidden)">Toggle reviewed photos / 顯示或隱藏已審閱照片</button><main>'''+''.join(cards)+'</main></html>'
    (directory/'gallery.html').write_text(page,encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    main(p.parse_args().directory.resolve())
