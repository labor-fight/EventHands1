#!/usr/bin/env python3
"""Original S43 template with small explanatory text removed at the user's request."""
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
from draw_s43_template_polished import draw, text, BLUE, TEAL

OUT = Path(__file__).resolve().parents[2] / "docs/assets/s43_w50_clean_20261019"


def main():
    fig=draw()
    ax=fig.axes[0]
    for item in list(ax.texts):
        if item.get_gid() not in {"module_label","figure_title","operator"}:
            item.remove()
        elif item.get_gid()=="module_label" and "查询时刻" in item.get_text():
            item.set_text("查询时刻 $t_k$：双极性 LNES\n"+r"$W=50$ ms · $h=50$ ms")
    text(ax,243.5,62,"W50",size=9.0,color=BLUE,ha="right")
    # Keep the essential timing labels at the same size as the module labels.
    text(ax,176.5,16.3,"W = 50 ms",size=8.2,color=BLUE)
    text(ax,212,16.3,"h = 50 ms",size=8.2,color=TEAL)
    for x,s in ((159,r"$t_0$"),(194,r"$t_k$"),(229,r"$t_{k+1}$")):
        text(ax,x,8,s,size=8.2,color=TEAL)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for item in ax.texts:
        b=item.get_window_extent(renderer)
        assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,item.get_text()
    OUT.mkdir(parents=True,exist_ok=True)
    stem=OUT/"s43_w50_clean"
    for ext in ("svg","pdf","png"):
        path=stem.with_suffix("."+ext)
        fig.savefig(path,dpi=600,facecolor="white")
        print(path)
    fig.savefig(stem.with_suffix(".tiff"),dpi=600,facecolor="white",pil_kwargs={"compression":"tiff_lzw"})
    with zipfile.ZipFile(OUT/"s43_w50_clean_pack.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.suffix!=".zip": z.write(p,p.name)
    plt.close(fig)


if __name__=="__main__":
    main()
