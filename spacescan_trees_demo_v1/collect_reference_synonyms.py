"""Additional licensed images using explicitly source-verified taxonomic synonyms."""
import json
from .collect_references import Client, collect, write_json, sheets, ROOT

OUT=ROOT/'external_datasets/spacescan_whole_tree_references_20260910_v2/mixed_v3/additional'
MAPPINGS={
    'Photinia benthamiana':('Pourthiaea pustulata','https://powo.science.kew.org/taxon/urn:lsid:ipni.org:names:727342-1'),
    'Thevetia peruviana':('Cascabela thevetia','https://powo.science.kew.org/taxon/urn:lsid:ipni.org:names:47927-2/general-information')}

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'images').mkdir(exist_ok=True)
    entries=json.loads((OUT/'manifest.json').read_text(encoding='utf-8')) if (OUT/'manifest.json').exists() else []
    errors=[]; client=Client(OUT);seen={str(r['photo_id']) for r in entries}
    for label,(query,source) in MAPPINGS.items():
        try:
            collect(client,label,[query],entries,seen,errors,per_class=6)
        except Exception as e:
            errors.append({'species':label,'error':str(e)})
        for r in entries:
            if r['species']==label:r['synonym_source']=source
        write_json(OUT/'manifest.json',entries)
        print(label,len([r for r in entries if r['species']==label]),flush=True)
    write_json(OUT/'errors.json',errors);sheets(OUT,entries)

if __name__=='__main__':main()
