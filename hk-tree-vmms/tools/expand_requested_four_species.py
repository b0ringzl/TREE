"""User-requested species pilot; Albizia lebbeck, NOT Acacia auriculiformis."""
import expand_hk_species_pilot as experiment

if __name__=='__main__':
    experiment.NAMES={'Delonix regia':'凤凰木','Aleurites moluccana':'石栗',
                      'Leucaena leucocephala':'银合欢','Albizia lebbeck':'大叶合欢'}
    experiment.OUT=experiment.DATA.parent/'species_expansion_requested_four_20260907'
    if (experiment.OUT/'results.json').exists():raise SystemExit('Existing results preserved; use a new version to rerun.')
    experiment.USE_FRAME_SOURCES=True
    experiment.LIMIT_PER_CLASS=3
    experiment.main()
