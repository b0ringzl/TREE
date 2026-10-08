"""Small second-photo search; original gallery and training files remain untouched."""
import json
from .collect_references import Client, collect, write_json, sheets, ROOT
from .collect_reference_synonyms import OUT

def main():
    entries=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    base=ROOT/'external_datasets/spacescan_whole_tree_references_20260910_v2'
    original=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    seen={str(r['photo_id']) for r in entries+original};client=Client(OUT);errors=[]
    for species in ['Adenanthera microsperma','Artocarpus heterophyllus','Crateva unilocularis','Elaeocarpus sylvestris','Photinia benthamiana']:
        try:collect(client,species,[species],entries,seen,errors,per_class=3)
        except Exception as e:errors.append({'species':species,'error':str(e)})
        print(species,len([r for r in entries if r['species']==species]),flush=True)
    write_json(OUT/'gap_errors.json',errors);sheets(OUT,[r for r in entries if r['species']!='Thevetia peruviana'])

if __name__=='__main__':main()
