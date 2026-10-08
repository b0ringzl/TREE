"""Compare frozen original BioCLIP2.5 and deployed H120 on identical images."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import fresh_bioclip146_benchmark_20261006 as base
from training_safety import verify_backup
OUT = ROOT / 'outputs/rare_species_fresh_validation_20261006_v1'

if __name__ == '__main__':
    verify_backup(OUT / 'evaluation_protocol_snapshot/pre_training_backup')
    base.summarize = lambda out: None
    base.evaluate(OUT)
