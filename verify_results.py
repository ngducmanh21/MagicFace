"""Create a visual AU verification report from already generated images."""

import argparse
import json
from pathlib import Path

from mgface.verification import add_report_arguments, report_from_args, write_verification_report


def load_saved_results(path):
    """Reuse scores and copied images from an earlier report, even after moving it."""
    path = Path(path).resolve()
    report = json.loads(path.read_text(encoding='utf-8'))
    cases = []
    scores = {'estimator': report['estimator'], 'version': report.get('estimator_version'),
              'device': report.get('estimator_device'), 'images': {}}
    for item in report['cases']:
        case = {key: item[key] for key in ('label', 'requested_aus', 'seed', 'inference_steps', 'fer') if key in item}
        for role in ('source', 'result'):
            case[role] = str((path.parent / item[f'{role}_asset']).resolve())
            score = item[f'{role}_score']
            if case[role] in scores['images'] and scores['images'][case[role]] != score:
                raise ValueError(f'Conflicting saved AU scores for {case[role]}')
            scores['images'][case[role]] = score
        cases.append(case)
    return report, cases, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--manifest', type=Path)
    inputs.add_argument('--results_json', type=Path,
                        help='Reuse existing scores and images without invoking LibreFace.')
    parser.add_argument('--output_dir', type=Path, required=True)
    add_report_arguments(parser)
    args = parser.parse_args()
    if args.results_json:
        previous, cases, scores = load_saved_results(args.results_json)
        report = write_verification_report(
            cases, args.output_dir, scores=scores,
            au_delta_scale=args.au_delta_scale if args.au_delta_scale is not None else previous['au_delta_scale'],
            title=args.report_title, metadata=previous.get('metadata', {}),
            evidence=args.evidence, figure_formats=args.figure_formats,
        )
        print(f'Report: {args.output_dir.resolve() / "report.html"}')
        print(f'Reused AU scores: {report["scored_cases"]}/{len(cases)} cases scored.')
        return
    manifest = json.loads(args.manifest.read_text())
    cases = manifest['cases']
    # Paths are relative to the manifest, never to the caller's working directory.
    for case in cases:
        for key in ('source', 'result'):
            case[key] = str((args.manifest.resolve().parent / case[key]).resolve())
    report_from_args(cases, args.output_dir, args, metadata=manifest.get('metadata', {}))


if __name__ == '__main__':
    main()
