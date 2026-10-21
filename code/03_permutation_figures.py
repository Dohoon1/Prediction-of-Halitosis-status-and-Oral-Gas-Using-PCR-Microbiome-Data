"""Render completed saved label-permutation results, with no model fitting."""
from pathlib import Path
import argparse
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def render(results, output, dpi=300):
    results, output = Path(results), Path(output)
    manifest = json.loads((results/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise ValueError('Permutation run is incomplete; final significance figures cannot be rendered.')
    base = results/'permutation'
    null = pd.read_csv(base/'maximum_statistic_null.csv')
    tests = pd.read_csv(base/'maximum_statistic_tests.csv')
    if not {'family','maximum_score'}.issubset(null) or not {'family','observed_maximum','p','status'}.issubset(tests):
        raise ValueError('Unexpected saved permutation result schema.')
    if tests.empty or not tests.status.eq('complete').all() or tests.p.isna().any():
        raise ValueError('A completed null and final p value are required for every reported family.')
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(math.ceil(len(tests)/2), 2, figsize=(12, 4*math.ceil(len(tests)/2)), squeeze=False)
    for ax, row in zip(axes.flat, tests.itertuples()):
        values = null.loc[null.family==row.family,'maximum_score']
        if values.empty:
            raise ValueError(f'Missing null scores for {row.family}')
        ax.hist(values, bins=min(30,max(3,int(len(values)**.5))), color='#7CA6BA', edgecolor='white')
        ax.axvline(row.observed_maximum,color='#C95B4C',lw=2,label='Observed maximum (same CV)')
        ax.set(title=f'{row.family}; family-wise p = {row.p:.3g}',xlabel='Saved maximum CV statistic',ylabel='Permutation count')
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(tests):]:
        ax.axis('off')
    fig.tight_layout()
    paths=[]
    for suffix in ['png','tiff']:
        path=output/f'Supplement_maximum_statistic_null.{suffix}'
        fig.savefig(path,dpi=dpi,bbox_inches='tight',**({'pil_kwargs':{'compression':'tiff_lzw'}} if suffix=='tiff' else {}))
        paths.append(str(path))
    plt.close(fig)
    caption=('Maximum-statistic label-permutation distributions for each prespecified candidate family. '
        'The observed reference and each null use the same CV repetitions, inner tuning and selection procedure. '
        'Classification folds are re-stratified using the permuted labels; gas vectors are permuted jointly. '
        'P values use (exceedances + 1)/(B + 1). Families and completed B are recorded in the saved analysis files. '
        'This tests global predictive association over the declared family; it is not a conditional test of PCR increment beyond clinical predictors. '
        'Development runs are marked in the source manifest and cannot be presented as final B=1000 inference.')
    (output/'permutation_figure_caption.txt').write_text(caption,encoding='utf-8')
    (output/'figure_manifest.json').write_text(json.dumps({'status':'complete','source_results':str(results.resolve()),'files':paths,'analysis_performed':False,'publication_run':manifest.get('publication_run')},indent=2),encoding='utf-8')
    return paths


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dpi',type=int,default=300)
    args=parser.parse_args()
    if args.results.resolve()==args.output.resolve() or args.dpi<1:
        parser.error('Use a separate output directory and positive dpi.')
    for path in render(args.results,args.output,args.dpi):
        print(path)
