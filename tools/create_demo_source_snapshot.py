"""Build an auditable code-only snapshot; never stage, commit, push or alter live files.

Used for the 2026-10-08 repository handoff. Human-readable reports and hashes
record omissions; the existing Git index is intentionally not used.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = [
    'spacescan_trees_demo_v1', 'tree_species_teacher_demo', 'tree_species_demo',
    'spacescan_android_sstv1', 'spacescan_android', 'hk-tree-vmms',
    'hk-tree-labeler', 'hk-tree-labeler-labelpaw-hover-sandbox', 'LabelPaw-web-images',
    'gis-map-preview-site', 'conservation_review', 'conservation_expansion_review',
    'tools', 'docs', 'Lidar', 'full_image_species_baseline',
    'image_species_baseline', 'nj_yolo11_vs_resnet50', 'yolo11_species_baseline',
]
SKIP_DIRS = {
    '.git', '.runtime', 'runtime', 'node_modules', '__pycache__', 'build', 'dist',
    '.gradle', 'logs', 'log', 'cache', 'view_cache', 'backups', 'models',
    'test_data', 'derived', 'outputs', 'output', 'experiments', 'runs', 'wandb',
    'checkpoints', 'samples', 'data', 'datasets', 'envs', 'reference_gallery',
    'pctrees_github', 'vendor', 'third_party',
}
TEXT_SUFFIXES = {
    '.py', '.js', '.cjs', '.mjs', '.ts', '.tsx', '.jsx', '.vue', '.html', '.css',
    '.json', '.md', '.txt', '.ps1', '.cmd', '.bat', '.gradle', '.xml', '.toml',
    '.yml', '.yaml', '.cfg', '.ini', '.properties', '.svg', '.csv',
}
EXTRA = [
    'AGENTS.md', 'requirements-demo.txt', 'START_SPACESCAN_TREES.cmd', 'START_SPACESCAN_IMAGE.cmd',
    'START_SPACESCAN_MULTIMODAL.cmd',
    'tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120/deployment.json',
    'tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120/labels.json',
    'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/deployment.json',
]
SECRET_PATTERNS = [
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})\b'),
    re.compile(r'https?://[^\s/:]+:[^\s/@]+@'),
    re.compile(r'(?i)(?:api[_-]?key|client[_-]?secret|access[_-]?token|password|accountkey)\s*[=:]\s*[\"\x27]([A-Za-z0-9_+/=-]{16,})[\"\x27]'),
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def selected_files():
    result = set()
    for folder in FOLDERS:
        start = ROOT / folder
        if not start.is_dir():
            continue
        for directory, names, filenames in os.walk(start):
            names[:] = [n for n in names if n not in SKIP_DIRS and not n.startswith(('.venv', '.cache', '.codex', '.tmp'))]
            for name in filenames:
                path = Path(directory) / name
                if name in {'local.properties', '.env', 'samples.json', 'SOURCE_MANIFEST.json'} or name.startswith('.env.'):
                    continue
                if path.suffix.lower() not in TEXT_SUFFIXES and name not in {'.gitignore', '.gitattributes', 'gradlew', 'Dockerfile', 'LICENSE', 'COPYING', 'NOTICE', 'AUTHORS'}:
                    continue
                if path.suffix.lower() == '.csv' and path.parent != ROOT / 'tools':
                    continue
                if path.stat().st_size <= 5 * 1024 * 1024:
                    result.add(path)
    result.update(ROOT / path for path in EXTRA if (ROOT / path).is_file())
    return sorted(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    records, findings, syntax_errors = [], [], []
    for source in selected_files():
        rel = source.relative_to(ROOT).as_posix()
        data = source.read_bytes()
        try:
            text = data.decode('utf-8-sig', errors='strict')
        except UnicodeDecodeError:
            text = data.decode('gb18030', errors='replace')
        for pattern in SECRET_PATTERNS:
            # Explicit non-credential fixture testing rejection of URL userinfo.
            fixture_paths = {'spacescan_trees_demo_v1/static/test-dropbox-import.cjs', 'tools/create_demo_source_snapshot.py'}
            scanned = text.replace('https://someone:password@www.dropbox.com/s/a', 'https://fixture.invalid/') if rel in fixture_paths else text
            if pattern.search(scanned):
                findings.append(rel)  # Never print credential values.
        if source.suffix == '.py':
            try:
                ast.parse(data, filename=rel)
            except SyntaxError as exc:
                syntax_errors.append({'path': rel, 'line': exc.lineno, 'error': exc.msg})
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target.read_bytes()) != digest(data):
            raise RuntimeError('Snapshot verification failed: ' + rel)
        records.append({'path': rel, 'bytes': len(data), 'sha256': digest(data)})
    # The original deleted README and existing staging area remain untouched.
    shutil.copy2(ROOT / 'docs/CURRENT_DEMOS_20261008.md', out / 'README.md')
    shutil.copy2(ROOT / 'docs/SOURCE_ASSETS_REQUIRED.md', out / 'ASSETS_REQUIRED.md')
    shutil.copy2(ROOT / 'docs/SOURCE_SNAPSHOT.gitignore', out / '.gitignore')
    shutil.copy2(ROOT / 'docs/SOURCE_SNAPSHOT_START.ps1', out / 'spacescan_trees_demo_v1/start-live.ps1')
    (out / 'tree_species_demo/samples.json').write_text('[]\n', encoding='utf-8')
    report_dir = out / 'docs/validation'
    report_dir.mkdir(exist_ok=True)
    reports = {
        'outputs/bioclip_flora_integration_20261008_v1/REPORT_ZH.md': 'world_model_20261008.md',
        'outputs/world_model_popup_20261008_v1/ACCEPTANCE_ZH.md': 'world_model_popup_20261008.md',
        'outputs/sstv1_ablation_summary_20261007_v2/REPORT_ZH.md': 'sstv1_ablation_20261007.md',
    }
    for source, name in reports.items():
        if (ROOT / source).is_file():
            shutil.copy2(ROOT / source, report_dir / name)
    # Audit the final payload too, including documentation and launch overrides.
    source_hashes = {r['path']: r['sha256'] for r in records}
    records = []
    for target in sorted(out.rglob('*')):
        if not target.is_file():
            continue
        rel = target.relative_to(out).as_posix()
        data = target.read_bytes()
        text = data.decode('utf-8-sig', errors='replace')
        if rel in {'spacescan_trees_demo_v1/static/test-dropbox-import.cjs', 'tools/create_demo_source_snapshot.py'}:
            text = text.replace('https://someone:password@www.dropbox.com/s/a', 'https://fixture.invalid/')
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            findings.append(rel)
        records.append({'path': rel, 'bytes': len(data), 'sha256': digest(data),
                        'original_source_sha256': source_hashes.get(rel)})
    (out / 'SNAPSHOT_AUDIT.json').write_text(json.dumps({
        'source_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'scope': 'Code-only source snapshot; no model/data/media/database credentials',
        'files': records, 'secret_findings_paths_only': sorted(set(findings)),
        'python_syntax_errors': syntax_errors,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'files': len(records), 'bytes': sum(r['bytes'] for r in records),
                      'secret_findings_paths_only': sorted(set(findings)),
                      'python_syntax_errors': syntax_errors}, ensure_ascii=False))
    if findings or syntax_errors:
        raise SystemExit('Snapshot requires review before publishing; no Git operation performed.')


if __name__ == '__main__':
    main()
