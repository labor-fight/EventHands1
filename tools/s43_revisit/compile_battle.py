"""Validate all 100 real cross-reviews and publish only decision-relevant disputes."""
from pathlib import Path
import collections
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/s43_revisit_20261028'
BATTLE = OUT/'battle'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cid(value):
    match = re.fullmatch(r'D(\d{2})[-_]?Q(\d{2})', value)
    if match is None:
        raise ValueError(value)
    return f'D{match[1]}Q{match[2]}'


def section(path, start, end, content):
    path = Path(path)
    text = path.read_text()
    block = start+'\n'+content+'\n'+end
    if start in text:
        a, b = text.index(start), text.index(end)+len(end)
        text = text[:a]+block+text[b:]
    else:
        text += '\n\n---\n\n'+block+'\n'
    path.write_text(text)


def main():
    r1, r2, source = {}, {}, {}
    for group in ('d01_d03','d04_d06','d07_d10'):
        one, two = (BATTLE/f'r{r}_{group}.json' for r in (1,2))
        a, b = json.loads(one.read_text()), json.loads(two.read_text())
        author = a.get('author', a.get('author_agent'))
        reviewer = b['reviewer']
        assert author and author != reviewer, (author,reviewer)
        recorded = b.get('source_sha256',b.get('reviewed_input_sha256',b.get('reviewed_source_sha256')))
        assert recorded == sha(one), f'R1 bytes changed after cross-review: {one}'
        source[group] = dict(R1=str(one.relative_to(ROOT)),R1_sha256=sha(one),
                             R2=str(two.relative_to(ROOT)),R2_sha256=sha(two),
                             author=author, reviewer=reviewer)
        for c in a['cells']:
            key = cid(c['id']); assert key not in r1
            for arm in 'ABC':
                for field in ('support','challenge','revision_or_test'):
                    assert isinstance(c[arm][field],str) and c[arm][field].strip(), (key,arm,field)
            r1[key] = c
        for c in b.get('reviews',b.get('cells')):
            key = cid(c['id']); assert key not in r2
            assert c['disposition'] in ('accept','revise')
            for field in ('critical_issue','required_change','decisive_test'):
                assert c[field].strip(), (key,field)
            r2[key] = c
    expected = {f'D{d:02d}Q{q:02d}' for d in range(1,11) for q in range(1,11)}
    assert set(r1) == set(r2) == expected
    counts = dict(collections.Counter(c['disposition'] for c in r2.values()))
    summary = dict(schema='s43_revisit_battle_compiled_v1', execution_complete=True,
                   perspectives=10, questions=10, R1_cells=100, candidate_reviews=300,
                   R2_cross_reviews=100, dispositions=counts, independent_humans=0,
                   actual_executors=['/root','/root/rel_core','/root/rel_geometry','/root/rel_integration'],
                   selection_by_vote=False, source=source,
                   normalized_cells={k:dict(R1=r1[k],R2=r2[k]) for k in sorted(expected)})
    (BATTLE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    names = ['事件物理','信息论','几何','运动学','概率推断','连续时间','非刚体跟踪','异步计算','训练优化','实验设计']
    matrix = ['| 专业视角 | '+' | '.join(f'Q{q:02d}' for q in range(1,11))+' |',
              '|---|'+'---|'*10]
    for d, name in enumerate(names,1):
        values = ['修订' if r2[f'D{d:02d}Q{q:02d}']['disposition']=='revise' else '保留' for q in range(1,11)]
        matrix.append(f'| D{d:02d} {name} | '+' | '.join(values)+' |')
    review = '''<a id="s43-third-round-100-20261028"></a>
### 第三轮（2026-10-28）：REL负结果后的100视角交叉审查

本节独立于前文第二轮，不覆盖旧审查或旧结果。实际执行者是根代理与三个子代理；100指10类专业×10个问题，**不是100名真人/独立模型/独立实验证据**。R1逐格同时评A原生时空场、B关节反证片段、C遮挡顺序，300项候选判断；R2由不同执行者逐格交叉审查100格，58保留、42需修订。修订数不是支持票，选型不按多数。

Q01信息单元/取舍，Q02因果/同刻，Q03极性/外观，Q04身份/遮挡，Q05关节证据/共模，Q06前态更正后历史，Q07事件/手态/query时钟，Q08梯度/学习/泄漏，Q09成本/有界性，Q10近邻/反例/决定性实验。

'''+'\n'.join(matrix)+'''

“保留”表示R1论证可保留，绝不表示候选通过实验。全部逐格原文、输入版本SHA、实际作者与跨审者见[机器可核100格](../outputs/s43_revisit_20261028/battle/summary.json)。[R1 D01—03](../outputs/s43_revisit_20261028/battle/r1_d01_d03.json)／[R2](../outputs/s43_revisit_20261028/battle/r2_d01_d03.json)，[R1 D04—06](../outputs/s43_revisit_20261028/battle/r1_d04_d06.json)／[R2](../outputs/s43_revisit_20261028/battle/r2_d04_d06.json)，[R1 D07—10](../outputs/s43_revisit_20261028/battle/r1_d07_d10.json)／[R2](../outputs/s43_revisit_20261028/battle/r2_d07_d10.json)。原R1的不同ID分隔符仅在合并索引中正规化，未改原文件字节。

#### 影响选择的争议与实质修订

| 争议 | 交叉审查入口 | 本轮处理 |
|---|---|---|
| 骨管/voxel/轨迹query是否新原理 | D02Q10、D03Q10、D07Q10 | 接受DCT/Taming/ETAP/EVDM近邻；A作为备选与强时间基线，只有解剖轨迹的独特任务区别可升级 |
| B是否只是top-K或coreset | D02Q09、D04Q09、D08Q05、D10Q09 | 保存原生完整片段而非分数，明确有限解剖族；先比uniform/density/recency/quota/通用task选择和同byte完整raw代价，不能只胜弱采样 |
| 错prior是否在选样前发生 | D07Q04、D03Q06 | 必须先错anchor再选择；字典冻结后只撤销注入的anchor/root误差，绝不给当前正确手指姿态 |
| uniform保底能否保证任意更正后仍充分 | D02Q06、D05Q06 | 不能；仅缓解选择偏差，未留历史不可恢复，过期/越族明确unknown或缺证据 |
| 缺测、未留和真实静默是否混淆 | D01Q03、D05Q01、D08Q04 | 完整片段/曝光/coverage单列；采样后的0不当物理负证据，粗支撑与fine内容分开 |
| oracle关系是否raw信息上界 | D02Q10、D05Q10 | GT ownership/visibility放特权旁支，只测假定关系的潜在效用；event-only提取另验，禁止画错误数据处理链 |
| 相同终点路径正控是否证明FK信息 | D04Q01、D04Q02 | 同终点仅作稳定性/时间利用负控；重建信息正控须不同当前FK目标，找不到物理匹配对记缺证据 |
| detach是否使读出不能纠错 | D03Q08、D09Q06 | detach只阻断特定坐标/跨态梯度，不阻止网络学如何使用合法历史；数值记忆和BPTT跨度分开 |
| 训练GT与部署oracle是否混同 | D04Q08、D09Q08 | 合法训练监督单列；构造/查询读当前GT才是部署泄漏，β0与外供β分报 |
| 重复源与真实同刻重数 | D05Q07 | 按source_range/ordinal/seq防重，不能按(x,y,t,p)把真实多次触发去重 |
| 不同query历史可否要求字典相同 | D06Q07、D08Q02 | pure-query审计固定commit与pose历史；不同反馈轨迹合法产生不同字典，不能误判不一致 |
| 同步数是否等价训练机会 | D09Q01、D09Q09 | 独立encoder需同encoder强sensor控制、可恢复FIT正控及合理输出预算；保留预训练收益与BN/优化因素 |
| 计算局部是否等于全系统异步 | D08Q01、D08Q09 | 逐事件、封槽选择、pose核更新、query网络与mesh分别计，候选bank及未选pool也入账 |
| RA是否纯关节、778是否面数 | D10Q05、D03Q09 | J0对齐仍含rootrotation，MPVPE仍V0非腕，778是顶点；局部运动另做解剖/去根诊断 |

本轮研究优先级为B关节反证片段，备选A完整时空场；不是B已通过。C缺可部署切换证据，保留条件研究。完整数学/张量/接口与杀点见[第三轮主方案](#s43-third-round-design-20261028)。三案都可能被原子实验否定，不因Nature目标降低门或强行大训。
'''
    # 2026-10-28: the two S43 design/review documents were merged into the research document
    # (附录 O / 附录 P, headings one level deeper, same markers); sections are rendered there.
    section(ROOT/'docs/网络结构与可观测性诊断研究.md',
            '<!-- S43_REVISIT_BATTLE_BEGIN -->','<!-- S43_REVISIT_BATTLE_END -->',review)
    spec = (OUT/'selected_prototype_spec.md').read_text()
    spec = re.sub(r'^(#{1,4}) ',r'##\1 ',spec,flags=re.M)
    selection = '''<a id="s43-third-round-design-20261028"></a>
### 第三轮：从REL失败归因到独立前端（2026-10-28）

前文A/B/C为第二轮历史定义；本节第三轮重新编号，不能混读。依据[归因实测](#s43-part-09)和[100视角两轮交叉审查](#s43-third-round-100-20261028)，本轮不要求保留A/Z/后绑定/Gram。

| 候选 | 表示对象与信息取舍 | MANO改变什么 | 处置 |
|---|---|---|---|
| A原生时空场 | 保联合xy-t-p有限场，丢有限窗外；先保观察后读轨迹 | 用合法解剖轨迹解释局部证据，不能只附joint标签 | 备选与强工程基线，已有直接轨迹query近邻 |
| B关节反证片段 | 保有限完整原生片段、coverage与选择版本，承认未选丢失 | 改变记忆分配对象，保护能反驳当前模式的局部时序 | 主原型优先级，待证明胜强简单选择与错prior条件 |
| C遮挡顺序 | 保出现/消失/邻接关系及null/多解释，不承诺材质身份 | 合法自遮挡转移约束关节构型 | 条件研究，先分清特权关系效用与event可提取性 |

[三案完整v0及各自反驳](../outputs/s43_revisit_20261028/candidates_v0.md)、[论文边界与原子试验](../outputs/s43_revisit_20261028/literature/candidate_principles.md)。下面是经关键跨审修订的可执行主原型定义；其创新句是待证命题，实施和实际裁决随后另记。

'''+spec
    backup=OUT/'backup_field_definition.md'
    if backup.exists():
        appendix=re.sub(r'^(#{1,4}) ',r'##\1 ',backup.read_text(),flags=re.M)
        selection+='\n\n'+appendix
    section(ROOT/'docs/网络结构与可观测性诊断研究.md',
            '<!-- S43_REVISIT_DESIGN_BEGIN -->','<!-- S43_REVISIT_DESIGN_END -->',selection)
    print(json.dumps(dict(complete=True,cells=len(expected),dispositions=counts,not_votes=True),ensure_ascii=False))


if __name__ == '__main__':
    main()
