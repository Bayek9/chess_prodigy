#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

PHOTO_MODELS = {
    'ft1b_screen': 'models/eval_suite_byfen/ft1b',
    'realv1_from_ft1b_e4ft1': 'models/eval_suite_byfen/realv1',
    'ftsafe_v2': 'models/eval_suite_byfen/ftsafe',
    'real_fen_all_train_v1_from_ftsafe_e1ft1': 'models/eval_suite_byfen/real_fen_all_train_v1_from_ftsafe_e1ft1',
    'real_fen_all_plus_pseudo5_v1_from_real_fen_all_e1': 'models/eval_suite_byfen/real_fen_all_plus_pseudo5_v1_from_real_fen_all_e1',
}
SCREEN_MODELS = {
    'ft1bis_screen': 'models/eval_suite_screen_gtv1/ft1bis/position_core_cases.csv',
    'ft1b_screen': 'models/eval_suite_screen_gtv1/ft1b/position_core_cases.csv',
    'realv1_from_ft1b_e4ft1': 'models/eval_suite_screen_gtv1/realv1/position_core_cases.csv',
    'ftsafe_v2': 'models/eval_suite_screen_gtv1/ftsafe/position_core_cases.csv',
    'real_fen_all_train_v1_from_ftsafe_e1ft1': 'models/eval_suite_screen_gtv1/real_fen_all_train_v1_from_ftsafe_e1ft1/position_core_cases.csv',
    'real_fen_all_plus_pseudo5_v1_from_real_fen_all_e1': 'models/eval_suite_screen_gtv1/real_fen_all_plus_pseudo5_v1_from_real_fen_all_e1/position_core_cases.csv',
}


def _metrics(csv_path: Path):
    df = pd.read_csv(csv_path)
    b = int(df['board_idx'].nunique())
    per = float((df['true'] == df['pred']).mean() * 100.0)
    avg = float((df['true'] != df['pred']).sum() / max(1, b))
    return round(per, 2), round(avg, 2), b


def main():
    rows = []
    for m, root in PHOTO_MODELS.items():
        vals = []
        for d in ('archive', 'samryan'):
            p = Path(root) / d / 'position_core_cases.csv'
            if not p.exists():
                continue
            per, avg, boards = _metrics(p)
            rows.append({'model': m, 'dataset': d, 'per_square': per, 'avg_wrong': avg, 'boards': boards})
            vals.append((per, avg))
        if len(vals) == 2:
            rows.append({'model': m, 'dataset': 'PHOTO_MEAN', 'per_square': round((vals[0][0] + vals[1][0]) / 2, 2), 'avg_wrong': round((vals[0][1] + vals[1][1]) / 2, 2), 'boards': None})

    for m, p in SCREEN_MODELS.items():
        p = Path(p)
        if not p.exists():
            continue
        per, avg, boards = _metrics(p)
        rows.append({'model': m, 'dataset': 'screen_gt_v1', 'per_square': per, 'avg_wrong': avg, 'boards': boards})

    out = pd.DataFrame(rows).sort_values(['dataset', 'avg_wrong', 'per_square'], ascending=[True, True, False])
    out_dir = Path('models/eval_suite_byfen')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'final_comparison_table.csv'
    out.to_csv(out_csv, index=False)

    photo = out[out['dataset'] == 'PHOTO_MEAN'].sort_values(['avg_wrong', 'per_square'], ascending=[True, False]).iloc[0]
    screen = out[out['dataset'] == 'screen_gt_v1'].sort_values(['avg_wrong', 'per_square'], ascending=[True, False]).iloc[0]

    summary = {
        'champion_photo_model': str(photo['model']),
        'champion_screen_model': str(screen['model']),
        'champion_photo_metrics': {'per_square': float(photo['per_square']), 'avg_wrong': float(photo['avg_wrong'])},
        'champion_screen_metrics': {'per_square': float(screen['per_square']), 'avg_wrong': float(screen['avg_wrong'])},
        'table_csv': str(out_csv.as_posix()),
    }
    (out_dir / 'final_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

