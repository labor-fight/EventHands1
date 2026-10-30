#!/usr/bin/env python3
"""Polish the original S43 horizontal-strip template, retaining the W50 raw flow.

Preserves the original layout: previous-state render branch above, sparse
network along the middle, asynchronous events -> SAE -> queried LNES below,
and the recurrent edge at the bottom.  W50 raw means no postfilter; it does
not remove the independent per-event SAE writer or the query clock.
Only presentation changes; no model or experiment files are modified.
"""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs/assets/s43_w50_sae_complete_20261019"
INK = "#27384A"
CAP = "#718094"
TEAL = "#188A77"
BLUE = "#3978AC"
ORANGE = "#D87739"
EDGE = "#D2DBE5"
PAPER = "#F7F9FC"


def text(ax,x,y,s,size=8.7,color=INK,ha="center",weight="normal",**kwargs):
    return ax.text(x,y,s,fontsize=size,color=color,ha=ha,va="center",fontweight=weight,zorder=12,**kwargs)


def line(ax,pts,color=TEAL,lw=1.05,head=True):
    for a,b in zip(pts[:-2],pts[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],c=color,lw=lw,zorder=3,solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle="-|>" if head else "-",
                                 mutation_scale=8,lw=lw,color=color,shrinkA=0,shrinkB=0,zorder=3))


def hand(ax,cx,cy,s,c):
    # Same line-icon idiom as the original figure, with anatomically ordered chains.
    chains = [((-0.18,-.32),(-.62,-.04),(-.91,.30)),
              ((-.30,-.05),(-.42,.56),(-.48,1.00)),
              ((-.06,.03),(-.08,.74),(-.07,1.22)),
              ((.20,.00),(.29,.66),(.36,1.06)),
              ((.43,-.12),(.66,.37),(.84,.72))]
    wrist=(cx,cy-.75*s)
    for chain in chains:
        pts=[wrist]+[(cx+x*s,cy+y*s) for x,y in chain]
        ax.plot([p[0] for p in pts],[p[1] for p in pts],c=c,lw=.8,solid_capstyle="round",zorder=7)
        for p in pts[1:]:
            ax.add_patch(Circle(p,.105*s,fc=c,ec="none",zorder=8))
    ax.add_patch(Circle(wrist,.15*s,fc=c,ec="none",zorder=8))


def grid(ax,cx,cy,s,c):
    for r in range(3):
        for k in range(3):
            ax.add_patch(Rectangle((cx+(k-1.5)*s*.63,cy+(r-1.5)*s*.63),s*.50,s*.50,
                                    ec=c,fc="none" if (r+k)%3 else c,lw=.65,alpha=.85,zorder=7))


def events(ax,cx,cy,s,c):
    pts=[(-.95,.32),(-.65,-.45),(-.48,.86),(-.10,-.16),(.16,.66),(.42,-.75),(.75,.35),(.98,-.22),(-.18,-.9)]
    for i,(x,y) in enumerate(pts):
        ax.add_patch(Circle((cx+x*s,cy+y*s),.13*s,fc=BLUE if i%2 else ORANGE,ec="none",zorder=7))


def planes(ax,cx,cy,s,c):
    for i in range(3):
        d=(i-1)*.30*s
        ax.add_patch(FancyBboxPatch((cx-.9*s+d,cy-.40*s-d),1.8*s,.8*s,
                      boxstyle="round,pad=0,rounding_size=.20",ec=c,fc="white",lw=.8,zorder=7+i))


def cube(ax,cx,cy,s,c):
    a=s*.72; d=s*.50
    front=[(cx-a,cy-a),(cx+a,cy-a),(cx+a,cy+a),(cx-a,cy+a)]
    ax.add_patch(Polygon(front,ec=c,fc="#FFF7EF",lw=.85,zorder=7))
    ax.add_patch(Polygon([front[2],front[3],(cx-a+d,cy+a+d),(cx+a+d,cy+a+d)],
                         ec=c,fc="#F9E7D9",lw=.85,zorder=7))
    ax.add_patch(Polygon([front[1],front[2],(cx+a+d,cy+a+d),(cx+a+d,cy-a+d)],
                         ec=c,fc="#F4D5BE",lw=.85,zorder=7))


def pool(ax,cx,cy,s,c):
    ax.add_patch(Polygon([(cx-.98*s,cy+.8*s),(cx+.98*s,cy+.8*s),(cx+.17*s,cy-.19*s),
                          (cx+.17*s,cy-.84*s),(cx-.17*s,cy-.84*s),(cx-.17*s,cy-.19*s)],
                         ec=c,fc="#FFF3E8",lw=.85,zorder=7))


def node(ax,x,y,w,h,label,icon=None,highlight=False,color=TEAL,size=8.8,caption=None):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=1.25",
                  fc="#FFF9F3" if highlight else "white",ec=ORANGE if highlight else EDGE,
                  lw=1.0 if highlight else .8,zorder=5))
    if icon:
        icon(ax,x+4.4,y+h/2,2.0,color)
        tx=x+7+(w-7)/2
        usable=w-9
    else:
        tx=x+w/2
        usable=w-3
    txt=text(ax,tx,y+h/2,label,size=size)
    txt.set_gid("module_label")
    if caption:
        text(ax,x+w/2,y-2.3,caption,size=6.8,color=CAP)
    return txt,usable


def draw():
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Noto Sans CJK JP","DejaVu Sans"],
                         "mathtext.fontset":"dejavusans","svg.fonttype":"none","pdf.fonttype":42,
                         "axes.unicode_minus":False})
    fig=plt.figure(figsize=(240/25.4,(66*240/248)/25.4),facecolor="white")
    ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,248),ylim=(0,66),aspect="equal"); ax.axis("off")
    ax.add_patch(FancyBboxPatch((.7,.8),246.6,64.3,boxstyle="round,pad=0,rounding_size=2.6",
                               fc=PAPER,ec="#DEE5ED",lw=.75,zorder=0))
    text(ax,4,62.0,"S43 稀疏渲染–比对手部跟踪",size=10.7,ha="left",weight="bold").set_gid("figure_title")
    text(ax,243.5,62.0,"W50 · 查询间隔 h = 50 ms（支持按需查询）",size=7.8,color=CAP,ha="right")
    text(ax,4,58.2,
         r"异步写入：$S[y_i,x_i,p_i]\leftarrow\max(S[y_i,x_i,p_i],t_i)$（O(1)/事件）；先接收 $t_i\leq t_k$ 再查询，时间戳为整数毫秒。",
         size=7.3,color=CAP,ha="left")
    text(ax,4,54.8,
         r"查询：$t_k=t_s+49+50k$ ms；$w_k=\min(50,t_k-t_s+1)$，$t_0=t_k-w_k+1$；$L_k=(S-t_0)/w_k$（窗内），其他为 0。",
         size=7.3,color=CAP,ha="left")
    ax.plot([4,244],[53.1,53.1],c="#E1E7EF",lw=.5,zorder=1)
    nodes=[]

    # The same upper row and left previous-state card as in the supplied figure.
    nodes.append(node(ax,4,24,18.5,10,"prev 51D\nMANO",hand,color=TEAL,size=8.4))
    nodes.append(node(ax,28,40,20,10,"MANO FK",hand,color=TEAL,size=8.7,caption="778 顶点 · 固定 β、K"))
    nodes.append(node(ax,53,40,29,10,"点溅射渲染\n占据 ∥ 逆深度",grid,color=TEAL,size=8.6,caption="240 × 180 · 2 通道"))
    line(ax,[(13.25,34),(13.25,45),(28,45)])
    line(ax,[(48,45),(53,45)])
    line(ax,[(82,45),(106,45),(106,34)])

    # The original top bypass goes to prev_mlp; its last branch supplies prev to the update.
    line(ax,[(13.25,45),(13.25,51.9),(235,51.9),(235,34)])
    line(ax,[(202,51.9),(202,50)])
    ax.add_patch(Circle((202,51.9),.38,fc=TEAL,ec="none",zorder=6))
    nodes.append(node(ax,188,40,28,10,"prev_mlp\n51 → 64 → 51",size=8.8))
    text(ax,233,51.0,"prev",size=7.0,color=TEAL,ha="right")

    # Unchanged middle-row organization, with lighter borders and disciplined captions.
    nodes.append(node(ax,90,24,31,10,"4 通道融合\n"+r"活跃位点 $A_0\ldots A_4$",grid,True,ORANGE,8.7))
    text(ax,96,21.0,"非零支持并集",size=6.4,color=CAP)
    nodes.append(node(ax,127,24,34,10,"稀疏 ResNet18-L3\n仅计算活跃位点",cube,True,ORANGE,8.6))
    text(ax,145,36.8,"64 / 128 / 256 · MaskedBN · CoordConv",size=6.7,color=CAP)
    nodes.append(node(ax,167,24,30,10,"活跃均值 ∥ 面积和\nFC 512 → 51",pool,True,ORANGE,8.6,caption="256 + 256 维读出"))
    line(ax,[(121,29),(127,29)])
    line(ax,[(161,29),(167,29)])
    line(ax,[(197,29),(201.1,29)])
    ax.add_patch(Circle((203,29),1.8,ec=TEAL,fc="white",lw=1.05,zorder=6))
    text(ax,203,29,"+",size=11,color=TEAL).set_gid("operator")
    line(ax,[(202,40),(203,30.8)])
    line(ax,[(204.8,29),(211,29)])
    text(ax,208,31.4,"Δ",size=8.0,color=TEAL).set_gid("operator")
    nodes.append(node(ax,211,24,32,10,"$x_k$ = prev + Δ\n零事件：保持前态",hand,color=TEAL,size=8.5,caption="3 + 3 + 45 维状态"))

    # Preserve the full original asynchronous input chain. Writer and query are independent.
    nodes.append(node(ax,28,9,27,10,"异步事件流\n"+r"$(x_i,y_i,t_i,p_i)$",events,color=BLUE,size=8.7))
    nodes.append(node(ax,61,9,33,10,"逐像素时间面 SAE\n每事件 O(1) 写入",planes,True,ORANGE,8.4,
                      caption="按像素 / 极性保存最近事件时刻"))
    nodes.append(node(ax,100,9,43,10,"查询时刻 $t_k$：双极性 LNES\n"+r"$L_k=(S-t_0)/w_k$（窗内）",
                      grid,True,ORANGE,8.3,caption="180 × 240 × 2；窗外 / 未写入位置为 0"))
    line(ax,[(55,14),(61,14)],color=BLUE)
    line(ax,[(94,14),(100,14)],color=BLUE)
    line(ax,[(121.5,19),(121.5,22.3),(114,22.3),(114,24)],color=BLUE)

    # A small timing key occupies the original unused lower-right space without changing the template.
    text(ax,194,18.2,"事件持续写入；查询时才执行稀疏前向",size=6.7,color=CAP)
    ax.add_patch(Rectangle((159,10.6),35,3.0,fc="#E4EEF7",ec="none",zorder=1))
    line(ax,[(154,12.1),(233,12.1)],color="#8798AA",lw=.65)
    for x,c in [(162,BLUE),(167,ORANGE),(175,BLUE),(178,ORANGE),(186,BLUE),(201,ORANGE),(210,BLUE),(219,BLUE)]:
        ax.add_patch(Circle((x,12.1),.32,fc=c,ec="none",zorder=6))
    for x in (159,194,229):
        ax.plot([x,x],[10.9,13.4],c=TEAL if x>159 else CAP,lw=.7,zorder=6)
    for x in (194,229):
        ax.add_patch(Circle((x,12.1),.62,fc=TEAL,ec="white",lw=.45,zorder=8))
    text(ax,176.5,15.4,"W = 50 ms",size=6.7,color=BLUE)
    text(ax,212,15.4,"h = 50 ms",size=6.7,color=TEAL)
    text(ax,159,8.5,r"$t_0=t_k-49$",size=6.5,color=CAP)
    text(ax,194,8.5,r"$t_k$",size=6.7,color=TEAL)
    text(ax,229,8.5,r"$t_{k+1}$",size=6.7,color=TEAL)

    # Exactly the long bottom return used in the supplied template, now direct W50 feedback.
    line(ax,[(243,29),(245.3,29),(245.3,3.8),(13.25,3.8),(13.25,24)],lw=1.05)
    text(ax,158,3.8,"$x_k$ 直接作为下一步 prev；空事件窗口将两路修正量同时置零。",size=7.2,color=TEAL,
         bbox=dict(fc=PAPER,ec="none",pad=1.0))

    # Fit labels inside the same original cards; do not let long text touch an icon or border.
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    scale=ax.transData.transform((1,0))[0]-ax.transData.transform((0,0))[0]
    for txt,width in nodes:
        while txt.get_window_extent(renderer).width > width*scale and txt.get_fontsize()>7.2:
            txt.set_fontsize(txt.get_fontsize()-.1)
            fig.canvas.draw()
    return fig


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    fig=draw(); fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for t in fig.axes[0].texts:
        b=t.get_window_extent(renderer)
        assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,t.get_text()
    base=OUT/"s43_w50_sae_complete"
    for ext in ("svg","pdf","png"):
        p=base.with_suffix("."+ext)
        fig.savefig(p,dpi=600,facecolor="white")
        print(p)
    fig.savefig(base.with_suffix(".tiff"),dpi=600,facecolor="white",pil_kwargs={"compression":"tiff_lzw"})
    (OUT/"figure_note.txt").write_text(
        "保留原图横向主链、上方前态渲染及MLP支路、下方异步事件→SAE→查询LNES支路与底部回环。\n"
        "SAE按像素和极性保存最近事件时间，整数毫秒；每事件O(1)更新S[y,x,p]←max(S[y,x,p],t_i)。查询t_k前仅摄入时间≤t_k的事件。\n"
        "t_s是当前有效段起点；k≥0且查询不超过段末。W50原始采样t_k=t_s+49+50k ms，窗口W=50ms，查询间隔h=50ms。\n"
        "一般按需查询的窗口宽w_k=min(50,t_k-t_s+1)，t_0=t_k-w_k+1；t_0≤S≤t_k时L_k=(S-t_0)/w_k，其余置0。正式50ms节拍查询w_k均为50ms。\n"
        "LNES中最新毫秒的最大值为49/50；窗首毫秒事件编码为0，不声称最新事件值为1。\n"
        "写入与查询独立；事件写入不触发神经网络，查询时执行完整稀疏前向。运行时支持其他查询时刻，图中明确使用W50/h50设定。\n"
        "上述定义由semkine/async_tracker.py的ingest/lnes与semkine/eval_track.py的W50采样核实；不将tools/s43/async_eval.py当前W200默认入口冒称为W50实测。\n"
        "W50原始流程不含后处理滤波；面积和指特征总和除以末层整张网格面积，零事件门对CNN和MLP合成的整份修正量生效。\n",encoding="utf-8")
    (OUT/"manifest.json").write_text(json.dumps({"template":"original horizontal strip, 248:66","protocol":"W50 raw with independent SAE writer and query","query_hop_ms":50,"window_ms":50,"first_query_offset_ms":49,"dpi":600,"formats":["SVG","PDF","PNG","TIFF"]},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with zipfile.ZipFile(OUT/"s43_w50_sae_complete_pack.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.suffix!=".zip": z.write(p,p.name)
    plt.close(fig)


if __name__=="__main__":
    main()
