#!/usr/bin/env python3
"""Stage Japanese runs into the unmodified FDB v1/v1.5 directory layout."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

def copy(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    if a.out_dir.exists() and a.overwrite: shutil.rmtree(a.out_dir)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for trial in sorted(path.parent for path in a.run_dir.glob('**/output.meta.json')):
        meta = json.loads((trial/'metadata.json').read_text(encoding='utf-8-sig'))
        task = meta['task']; target = a.out_dir/task/f"{meta['id']}_seed_{trial.name.removeprefix('seed_')}"
        target.mkdir(parents=True, exist_ok=True)
        for name in ('input.wav','clean_input.wav','output.wav','clean_output.wav','input.json','clean_input.json','output.json','clean_output.json'):
            copy(trial/name, target/name)
        events = meta.get('events', {})
        interval = (events.get('overlap') or events.get('interruption') or [[0.0, 0.0]])[0]
        official_meta = {'context_text':'', 'current_turn_text':'', 'timestamps': interval}
        (target/'metadata.json').write_text(json.dumps(official_meta, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        if task == 'smooth_turn_taking':
            end = max((x['end_sec'] for x in meta.get('user_segments',[])), default=0.0)
            (target/'turn_taking.json').write_text(json.dumps([{'timestamp':[end,end]}])+'\n')
        if task == 'user_interruption':
            text = ''.join(x.get('text','') for x in meta.get('user_segments',[]) if x.get('kind') == 'interrupt')
            (target/'interrupt.json').write_text(json.dumps([{'context':'','interrupt':text,'timestamp':interval}], ensure_ascii=False)+'\n', encoding='utf-8')
    return 0
if __name__ == '__main__': raise SystemExit(main())
