import json,sys
from pathlib import Path
from collect_sstv1_incremental import get_json,save,ROOT

def main():
    data=ROOT/'external_datasets/aquilaria_specialist_20260914_v1'
    source=ROOT/'outputs/distillation/20260908_0045'
    groups={}
    for phase in ('train','planning','selection','test'):
        for r in json.loads((source/f'{phase}_rows.json').read_text(encoding='utf8')):
            if r['species']=='Aquilaria sinensis' and 'observation:' in r['group_id']:
                oid=r['group_id'].split('observation:')[-1];groups.setdefault(oid,[]).append(phase)
    result=get_json('observations',{'id':','.join(groups),'per_page':200})
    save(data/'source_responses/old_aquilaria_observations.json',result)
    audit=[{'observation':o['id'],'observer':o['user']['id'],'phases':sorted(set(groups.get(str(o['id']),[]))),
        'current_name':o.get('taxon',{}).get('name'),'quality_grade':o['quality_grade']} for o in result['results']]
    save(data/'old_author_audit.json',audit)
    print('RETURNED',len(audit),'of',len(groups))
    new=json.loads((data/'manifest.json').read_text(encoding='utf8'))
    for author in sorted({r['observer_id'] for r in new}):
        print(author,sorted({p for r in audit if r['observer']==author for p in r['phases']}))
if __name__=='__main__':main()
