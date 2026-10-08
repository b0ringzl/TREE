"""Post-lock development diagnostics for predeclared last steps, never reselects."""
from pathlib import Path
import torch
import numpy as np
from distillation_core import read_json,save_json
from train_artocarpus_user_20260918 import heads_from_deployment,predict
from train_aq_ar_diagnostic_20260919 import evaluate_dev,unchanged
from aq_ar_dualview_next_20260919 import load_pair,tensor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/SSTv1_Aquilaria_UserCrops_20260919_v2'


def main():
    assert (OUT/'selection_lock.json').exists() and not (OUT/'last_steps_single_dev.json').exists()
    protocol=read_json(OUT/'protocol.json');unchanged(protocol);torch.set_num_threads(4)
    dev=load_pair(OUT,'dev')
    with np.load(OUT/'train_features.npz') as f:
        params=dict(groups=read_json(OUT/'classes.json')['groups'],temp=protocol['temperature'],extra=tensor(f['extra_bank']),scale=float(f['scale']))
    result={}
    for recipe in protocol['recipes']:
        heads=heads_from_deployment(protocol)
        heads.load_state_dict(torch.load(OUT/recipe['name']/'step_120.pt',map_location='cpu',weights_only=False)['model_state'])
        pred=predict(heads,dev[:3],params)
        result[recipe['name']]=evaluate_dev(dev[0],pred)
    unchanged(protocol)
    save_json(OUT/'last_steps_single_dev.json',dict(note='Fixed final-step diagnostics on development, selection already locked and not revised; no test touched.',results=result))
    print({k:v['combined'] for k,v in result.items()},flush=True)


if __name__=='__main__':main()
