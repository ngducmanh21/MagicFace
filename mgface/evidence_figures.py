"""Static AU/FER figures suitable for sharing or inclusion in a paper."""

from pathlib import Path
import textwrap


def render_evidence_figures(report, summary, output_dir, formats):
    import matplotlib
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    formats = tuple(dict.fromkeys(formats))
    if not formats or set(formats) - {'png', 'svg', 'pdf'}:
        raise ValueError('Figure formats must be a non-empty subset of png, svg, pdf.')
    directory = Path(output_dir) / 'figures'
    directory.mkdir(exist_ok=True)
    figures = []

    def new_figure(title, subtitle, size=(12, 5.5)):
        fig = Figure(figsize=size, facecolor='white')
        FigureCanvasAgg(fig)
        fig.suptitle(title, fontsize=17, fontweight='bold', x=0.07, ha='left', y=0.98)
        fig.text(0.07, 0.925, subtitle, fontsize=10, color='#516078', va='top')
        fig.text(0.07, 0.015, textwrap.shorten(report['title'], width=120, placeholder='...'),
                 fontsize=8, color='#64748b')
        return fig

    def save(fig, stem, title):
        files = []
        # Always provide a PNG thumbnail for the HTML, even for PDF/SVG-only export.
        for extension in dict.fromkeys(('png', *formats)):
            path = directory / f'{stem}.{extension}'
            fig.savefig(path, format=extension, dpi=180, facecolor='white')
            files.append(str(path.relative_to(output_dir)))
        figures.append({'title': title, 'files': files, 'preview': f'figures/{stem}.png'})
        fig.clear()

    def style(ax):
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.18)
        ax.set_axisbelow(True)

    with matplotlib.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                                'axes.labelcolor': '#334155', 'text.color': '#17243b',
                                'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        n = summary['n_scored_pairs']
        fig = new_figure('AU intensity before and after editing',
                         f'{n}/{summary["n_cases"]} scored pairs. Means are edit-case weighted; repeated sources count for each edit.')
        ax = fig.subplots()
        fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.15)
        if n:
            x = np.arange(len(summary['au']))
            ax.bar(x - 0.19, [r['source_mean'] for r in summary['au']], width=0.38,
                   label='Source', color='#94a3b8')
            ax.bar(x + 0.19, [r['result_mean'] for r in summary['au']], width=0.38,
                   label='Result', color='#0d9488')
            ax.set_xticks(x, [r['au'] for r in summary['au']])
            ax.set_ylabel('Mean measured intensity (0-5)')
            ax.set_ylim(0, 5)
            ax.legend(frameon=False)
            style(ax)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'AU measurements unavailable\nNo mean values have been estimated.',
                    ha='center', va='center', transform=ax.transAxes, fontsize=15)
        save(fig, 'fig_au_intensity', 'AU intensity: source vs result')

        names = list(dict.fromkeys(r['au'] for r in summary['au_control_response']))
        columns = min(3, len(names))
        rows = (len(names) + columns - 1) // columns
        fig = new_figure('Requested AU controls and measured response',
                         'Grouped by AU/control value, pooling combination edits. Bars = sample SD when n >= 2; labels = scored/total.',
                         size=(12, 3.2 * rows + 1.5))
        axes = fig.subplots(rows, columns, squeeze=False)
        fig.subplots_adjust(left=0.08, right=0.97, top=0.79 if rows == 1 else 0.86,
                            bottom=0.16 if rows == 1 else 0.08, hspace=0.55, wspace=0.32)
        for ax in axes.flat:
            ax.set_visible(False)
        for ax, name in zip(axes.flat, names):
            ax.set_visible(True)
            group = [r for r in summary['au_control_response'] if r['au'] == name]
            measured = [r for r in group if r['n_scored']]
            ax.set_title(name, fontweight='bold')
            ax.set_xlabel('Requested change (model units)')
            ax.set_ylabel('Measured change (intensity units)')
            ax.axhline(0, color='#cbd5e1', lw=1)
            all_x = [r['requested_delta'] for r in group]
            padding = max((max(all_x) - min(all_x)) * 0.15, 0.2)
            ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
            if measured:
                # A missing control is a gap, not an interpolated measurement.
                ax.plot(all_x, [r['change_mean'] if r['n_scored'] else np.nan for r in group],
                        'o-', color='#0d9488', label='Measured')
                for row in measured:
                    if row['change_std'] is not None:
                        ax.errorbar(row['requested_delta'], row['change_mean'], yerr=row['change_std'],
                                    color='#0d9488', capsize=4)
                    ax.annotate(f"{row['n_scored']}/{row['n_total']}",
                                (row['requested_delta'], row['change_mean']), xytext=(5, 9),
                                textcoords='offset points', fontsize=8)
                if report['au_delta_scale'] is not None:
                    ax.plot(all_x, [r['expected_delta'] for r in group], '--x', color='#b45309', label='Expected')
                ax.legend(fontsize=8, frameon=False)
                ax.margins(x=0.2, y=0.35)
            else:
                ax.text(0.5, 0.5, 'No scored pairs', transform=ax.transAxes, ha='center')
            for row in group:
                if not row['n_scored']:
                    ax.text(row['requested_delta'], 0.04, f"N/A\n0/{row['n_total']}",
                            transform=ax.get_xaxis_transform(), ha='center', fontsize=8, color='#64748b')
            style(ax)
        save(fig, 'fig_au_control_response', 'AU control response')

        for start in range(0, len(report['cases']), 24):
            cases = report['cases'][start:start + 24]
            height = max(5.5, 2.4 + len(cases) * 0.38)
            fig = new_figure('Measured AU changes per edit',
                             'Change = result - source (0-5 intensity scale). Outlined cells = requested non-zero edits. N/A = unscored.',
                             size=(14, height))
            ax = fig.subplots()
            fig.subplots_adjust(left=0.31, right=0.93, top=0.83, bottom=0.12)
            data = np.array([[r['measured_delta'] if r['measured_delta'] is not None else np.nan
                              for r in case['au_rows']] for case in cases])
            cmap = matplotlib.colormaps['RdBu_r'].copy()
            cmap.set_bad('#eef2f7')
            image = ax.imshow(np.ma.masked_invalid(data), vmin=-5, vmax=5, cmap=cmap, aspect='auto')
            ax.set_xticks(range(12), [r['au'] for r in cases[0]['au_rows']])
            ax.set_yticks(range(len(cases)), [f'{start + i + 1:03d}  ' +
                         textwrap.shorten(c['label'], width=38, placeholder='...') for i, c in enumerate(cases)])
            for i, case in enumerate(cases):
                for j, row in enumerate(case['au_rows']):
                    value = row['measured_delta']
                    color = 'white' if value is not None and abs(value) > 3 else '#334155'
                    ax.text(j, i, 'N/A' if value is None else f'{value:+.2f}', ha='center', va='center',
                            fontsize=8, color=color)
                    if row['requested_delta'] != 0:
                        ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94,
                                               fill=False, edgecolor='#17243b', linewidth=1.3))
            fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025, label='Measured change')
            save(fig, f'fig_au_changes_{start // 24 + 1:03d}', 'AU changes per edit')

        for role, metrics in summary['fer'].items():
            if metrics['status'] != 'available':
                continue
            labels = metrics['labels']
            count = len(labels)
            height = max(6.5, 3.2 + count * 0.42)
            fig = new_figure(f'FER confusion matrix - {role}',
                             f"{metrics['n_evaluated']} unique images evaluated; {metrics['n_missing_labels']} missing labels. "
                             f"Accuracy {metrics['accuracy']:.3f} | Macro F1 {metrics['macro_f1']:.3f}. Rows=true, columns=predicted.",
                             size=(max(13, count * 1.25 + 4), height))
            axes = fig.subplots(1, 2)
            fig.subplots_adjust(left=0.12, right=0.98, top=0.80, bottom=0.24, wspace=0.42)
            wrapped = ['\n'.join(textwrap.wrap(label, width=14)) for label in labels]
            for ax, key, title in zip(axes, ('confusion_counts', 'confusion_row_normalized'),
                                      ('Counts', 'Row-normalized recall')):
                values = np.array([[np.nan if v is None else v for v in row] for row in metrics[key]])
                vmax = max(1, int(values.max())) if key == 'confusion_counts' else 1
                cmap = matplotlib.colormaps['Blues'].copy()
                cmap.set_bad('#eef2f7')
                ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=vmax, cmap=cmap)
                ax.set_title(title)
                ax.set_xticks(range(count), wrapped, rotation=35, ha='right', fontsize=9)
                ax.set_yticks(range(count), wrapped, fontsize=9)
                ax.set_xlabel('Predicted class')
                ax.set_ylabel('True class')
                for i in range(count):
                    for j in range(count):
                        value = values[i, j]
                        label = ('N/A' if np.isnan(value) else
                                 (str(int(value)) if key == 'confusion_counts' else f'{value:.2f}'))
                        ax.text(j, i, label, ha='center', va='center', fontsize=9,
                                color='white' if value > vmax * 0.55 else '#17243b')
            save(fig, f'fig_fer_{role}_confusion', f'FER confusion matrix: {role}')
    return figures
