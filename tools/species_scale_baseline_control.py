"""Validation-only control: limit displayed labels without refitting the encoder."""
import sys
from pathlib import Path
import numpy as np
import torch
from species_scale_experiment import (BASE,load_json,write_json,original_probabilities,
                                     collapsed_baseline,tune_gate,metrics,targets_met)

def main(out):
    out=Path(out);protocol=load_json(out/'protocol.json');sets=load_json(out/'nested_class_sets.json')
    names=load_json(BASE/'datasets/full/dataset_summary.json')['class_names']
    original=torch.load(out/'original_linear.pt',map_location='cpu',weights_only=False)
    rows={s:load_json(out/(s+'_rows.json')) for s in ('planning','selection')}
    probabilities={s:original_probabilities(np.load(out/(s+'_features.npy')),original) for s in rows}
    result=[]
    for size,labels in sets.items():
        k=int(size);mapping={n:i for i,n in enumerate(labels)}
        y={s:np.array([mapping.get(r['label'],k) for r in rows[s]]) for s in rows}
        p={s:collapsed_baseline(probabilities[s],names,labels) for s in rows}
        gate=tune_gate(y['planning'],p['planning'],k,protocol)
        m=metrics(y['selection'],p['selection'],k,gate['threshold'],gate['margin'])
        result.append({'k':k,'planning_gate':gate,'selection':m,'target_met':targets_met(m,protocol)})
    write_json(out/'masked_baseline_validation.json',result)
    print([{r['k']:r['selection']} for r in result])

if __name__=='__main__':
    torch.set_num_threads(4);main(Path(sys.argv[1]))
