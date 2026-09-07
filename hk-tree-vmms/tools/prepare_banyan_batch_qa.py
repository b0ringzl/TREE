"""Generate numbered visual QA contact sheets; no automatic human acceptance."""
from PIL import Image, ImageDraw
from audit_hk_metric_height import DATA,read,write

OUT=DATA/'banyan_tracking_batch_v2'

def main():
    records=read(OUT/'results.json'); source={r['id']:r for r in read(DATA/'manifest.json')}
    qa={}; eligible=[]
    for r in records:
        if r['status']!='candidate_requires_review':continue
        reasons=[]
        if r['tracking']['observed_stem_bin_coverage']<.85:reasons.append('冠下高度层支持不足')
        if r['base_gap_to_roi_m']>.8:reasons.append('低位缺口较大，根颈待核准')
        if r['new_trunk_points']<r['old_trunk_points']*1.2:reasons.append('未显示足够的新增结构支持')
        qa[r['id']]={'disposition':'needs_rework' if reasons else 'awaiting_visual_review',
                     'note':'；'.join(reasons) if reasons else '已通过数值初筛，尚未投影预检，不等同可用树干'}
        if not reasons:eligible.append(r)
    # One view per existing inferred group for first review; alternatives retained.
    eligible.sort(key=lambda r:(-r['tracking']['observed_stem_bin_coverage'],r['id']))
    chosen=[];seen=set()
    for r in eligible:
        group=source[r['id']].get('tree_group',r['id'])
        if group in seen:
            qa[r['id']]={'disposition':'alternate_view','note':'同组已有优先预检视角，暂不重复交验；同木关系仍待确认。'}
        else:chosen.append(r);seen.add(group)
    write(OUT/'agent_visual_qa.json',qa)
    write(OUT/'visual_queue.json',[{'index':i+1,'id':r['id']} for i,r in enumerate(chosen)])
    for offset in range(0,len(chosen),6):
        sheet=Image.new('RGB',(1536,1500),'#f2f5ed');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(chosen[offset:offset+6]):
            x=(j%2)*768;y=(j//2)*500
            draw.text((x+8,y+5),str(offset+j+1)+' '+r['id'],fill='black')
            with Image.open(OUT/r['id']/'projection.jpg') as im:
                im.thumbnail((768,384));sheet.paste(im,(x,y+25))
            with Image.open(OUT/r['id']/'after.jpg') as im:
                im=im.resize((768,90));sheet.paste(im,(x,y+409))
        sheet.save(OUT/f'qa_sheet_{offset//6+1}.jpg',quality=95)
    print('Processed',len(records),'numerical candidates',len(eligible),'unique-group visual queue',len(chosen),flush=True)

if __name__=='__main__':main()
