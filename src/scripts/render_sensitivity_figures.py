"""Render sensitivity figures from the published aggregate tables."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'docs/assets/spod-dr/experiment-data'
OUT = ROOT / 'docs/assets/spod-dr/experiments'
COHORTS = ('DDR', 'IDRiD', 'MESSIDOR-2')
MODELS = [('EfficientNet-B0', 'B0', '#496b7c', 'o'),
          ('EfficientNet-B3', 'B3', '#a15233', 's'),
          ('ConvNeXt-Tiny', 'ConvNeXt', '#68764a', '^'),
          ('ResNet-50', 'ResNet-50', '#7f7281', 'D'),
          ('Swin-Tiny', 'Swin', '#247b79', 'P')]
STYLE = {'font.family': 'Times New Roman', 'font.size': 11,
         'text.color': '#222b2d', 'axes.labelcolor': '#222b2d',
         'axes.edgecolor': '#b4c3c8', 'xtick.color': '#656b6e',
         'ytick.color': '#656b6e', 'axes.spines.top': False,
         'axes.spines.right': False, 'svg.fonttype': 'none',
         'pdf.fonttype': 42, 'savefig.facecolor': 'white'}


def read(name):
    with (DATA / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def legend(fig, y=.995):
    handles = [Line2D([], [], color=c, marker=m, markerfacecolor='white',
                      linewidth=1.8, markersize=6, label=label)
               for _, label, c, m in MODELS]
    fig.legend(handles=handles, ncol=5, frameon=False,
               loc='upper center', bbox_to_anchor=(.53, y))


def save(fig, name):
    for fmt in ('png', 'svg'):
        path = OUT / (name + '.' + fmt)
        fig.savefig(path, dpi=240)
        if fmt == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def response(rows, axis, name, portrait=False):
    if portrait:
        fig, axs = plt.subplots(3, 2, figsize=(8, 8.5), sharey='col')
        fig.subplots_adjust(left=.10, right=.97, bottom=.10, top=.88, wspace=.25, hspace=.43)
    else:
        fig, axs = plt.subplots(2, 3, figsize=(12, 6.2), sharey='row')
        fig.subplots_adjust(left=.08, right=.98, bottom=.19, top=.78, wspace=.22, hspace=.24)
    grid = [.00625, .0125, .025, .05, .1] if axis == 'rho' else [2, 4, 6, 8, 10]
    for j, cohort in enumerate(COHORTS):
        if not portrait:
            axs[0,j].set_title(cohort, fontweight='bold', fontsize=15, pad=15)
        for i, metric in enumerate(('FPR_reported_pct', 'FNR_reported_pct')):
            ax = axs[j,i] if portrait else axs[i,j]
            if portrait:
                ax.set_title(cohort, fontweight='bold', fontsize=13, pad=8)
            ax.grid(axis='y', color='#e9e7e2', linewidth=.65)
            ax.set_axisbelow(True)
            for model, label, color, marker in MODELS:
                part = sorted([r for r in rows if r['dataset']==cohort and r['backbone']==model],
                              key=lambda r:float(r['setting']))
                assert len(part)==5, (model, cohort)
                ax.plot([float(r['setting']) for r in part],
                        [float(r[metric]) for r in part],
                        color=color, marker=marker, markerfacecolor='white',
                        markeredgewidth=1.4, linewidth=1.8, markersize=7)
            if axis=='rho': ax.set_xscale('log', base=2)
            ax.axvline(.025 if axis=='rho' else 6, color='#b4c3c8', linestyle='--', linewidth=1)
            ax.set_xticks(grid, [f'{v:g}' for v in grid])
            ax.tick_params(axis='x', labelbottom=(j==2 if portrait else i==1), rotation=35 if axis=='rho' else 0)
            ax.set_ylim(-.2 if i else 0, 8.6 if i else 85)
            if portrait or j==0: ax.set_ylabel('FNR (%)' if i else 'FPR (%)', fontsize=13)
    fig.supxlabel('SAM radius (ρ)' if axis=='rho' else 'Number of local queries (K)', y=.025, fontsize=14)
    legend(fig); save(fig,name)


def tradeoffs(rows):
    for cohort in COHORTS:
        fig, ax = plt.subplots(figsize=(9,5.7))
        fig.subplots_adjust(left=.1,right=.97,bottom=.14,top=.76)
        fig.suptitle(cohort,x=.1,ha='left',fontweight='bold',fontsize=18,y=.98)
        ax.grid(color='#e9e7e2',linewidth=.65);ax.set_axisbelow(True)
        labels=[]
        for model, label, color, marker in MODELS:
            part=[r for r in rows if r['dataset']==cohort and r['backbone']==model]
            assert len(part)==5
            for r in part:
                k=int(r['setting']);x=float(r['FPR_reported_pct']);y=float(r['FNR_reported_pct'])
                ax.scatter(x,y,s=65 if k==6 else 43,marker=marker,
                           facecolors=color if k==6 else 'white',edgecolors=color,linewidths=1.6)
                labels.append((x,y,str(k),color,k==6))
        ax.margins(x=.12,y=.24)
        ax.set_xlabel('False referrals (%)',fontsize=13)
        ax.set_ylabel('Missed referrals (%)',fontsize=13)
        fig.canvas.draw()
        occupied=[]
        for x,y,label,color,default in labels:
            for dx,dy in [(0,10),(12,7),(-12,7),(0,-15),(18,-9),(-18,-9),(24,15),(-24,15)]:
                a=ax.annotate(label,(x,y),xytext=(dx,dy),textcoords='offset points',
                              ha='center',color=color,fontweight='bold' if default else 'normal',fontsize=10)
                box=a.get_window_extent(fig.canvas.get_renderer()).expanded(1.15,1.15)
                if not any(box.overlaps(b) for b in occupied):break
                a.remove()
            else:
                a=ax.annotate(label,(x,y),xytext=(0,25),textcoords='offset points',ha='center',color=color,fontsize=10)
                box=a.get_window_extent(fig.canvas.get_renderer())
            occupied.append(box)
        legend(fig, y=.91);save(fig,'K_'+cohort)


def main():
    with plt.rc_context(STYLE):
        response(read('rho_S3.csv'),'rho','rho_response')
        tradeoffs(read('K_S4.csv'))
        response(read('K_S4.csv'),'K','K_response', portrait=True)
        response(read('rho_S3.csv'),'rho','rho_response_supplement', portrait=True)


if __name__=='__main__':
    main()
