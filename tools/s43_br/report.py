"""Build a reviewable small-experiment report without creating canonical rows."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import datetime
import numpy as np

from .experiment import OUT,ROOT,write_json,sha


def load(name): return json.loads((OUT/name).read_text())


def main():
    sources={
      'synthetic/root_x':('results/synthetic_root_x_test_final.json','test'),
      'synthetic/index_y':('results/synthetic_index_y_test_final.json','test'),
      'existing_SELECT/root_x':('results/existing_root_x_selection_v2.json','validation'),
      'existing_SELECT/index_y':('results/existing_index_y_selection_v2.json','validation'),
      'zgz_report/root_x':('results/zgz_root_x_report_final.json','test'),
      'zgz_report/index_y':('results/zgz_index_y_report_final.json','test')}
    reports={scope:load(path) for scope,(path,_) in sources.items()}
    rows=[]
    for scope,r in reports.items():
      split=sources[scope][1]
      for reader in ['common','adapted']:
       for arm in ['lnes','enhanced','br']:
        m=r[reader][arm][split]
        rows.append(dict(scope=scope,reader=reader,arm=arm,n=m['n'],accuracy=m['accuracy'],
                    absolute_joint_mm=m['absolute_joint_mm'],root_aligned_joint_mm=m['root_aligned_joint_mm'],
                    nll=m['nll'],brier=m['brier'],oracle_candidate_mm=m['oracle_candidate_mm']))
    csv_path=OUT/'mechanism_results.csv'
    if csv_path.exists(): raise FileExistsError(csv_path)
    with csv_path.open('w') as f:
      writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={scope:{'source':sources[scope][0],'metrics':[row for row in rows if row['scope']==scope],
                 'contrasts':r['contrasts'],'analytic':{key:value[sources[scope][1]] for key,value in r['analytic'].items()}}
             for scope,r in reports.items()}
    write_json(OUT/'results_summary.json',summary)
    prereg=load('protocol.json'); calibration=load('calibration_v2.json'); lock=load('final_lock.json')
    assert sha(ROOT/prereg['preregistration'])==prereg['prereg_sha256']
    assert sha(OUT/'calibration_v2.json')==lock['calibration_sha256']
    baseline=load('sources/synthetic_root_x.json')['frozen_production']
    integrity={p:{'expected':value,'actual':sha(ROOT/p),'unchanged':value==sha(ROOT/p)} for p,value in baseline.items()}
    assert all(v['unchanged'] for v in integrity.values())
    gate=[]
    for axis in ['root_x','index_y']:
      r=reports[f'existing_SELECT/{axis}']; s='validation'
      for reader in ['common','adapted']:
       b=r[reader]['br'][s]; simple=r[reader]['enhanced'][s]
       contrast=r['contrasts'][s][reader]
       accuracy_delta=b['accuracy']-simple['accuracy']
       relative_risk=1-b['absolute_joint_mm']/simple['absolute_joint_mm']
       positive=(accuracy_delta>=.05 and contrast['accuracy_br_minus_enhanced']['ci95'][0]>0) or (
             relative_risk>=.1 and contrast['absolute_mm_br_minus_enhanced']['ci95'][1]<0)
       gate.append({'axis':axis,'reader':reader,'accuracy_delta':accuracy_delta,
                    'relative_absolute_risk_reduction':relative_risk,'continue_gate':bool(positive)})
    decision=load('pre_zgz_decision.json')
    decision.update(at_final=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        SELECT_gates=gate,all_SELECT_gates_fail=not any(g['continue_gate'] for g in gate),
        representation_contracts='31 unit checks + 11 data checks + actual 2 FIT prefix equivalence',
        scientific_corrections=1,original_models_unchanged=True,
        failure_attribution={
          'representation_implementation':'validated finite model; overflow cancellation fixed before valid outcome inspection',
          'readout_adaptation':'shared kernel and interface-aware heads both tested; synthetic positive control succeeds',
          'optimization':'training fit/loss recorded; no evidence increasing steps is the targeted fix',
          'data_evidence':'chosen coarse joint slice often ambiguous; no cheap-code collision necessity established',
          'observation_model_and_coverage':'existing scores badly calibrated; finite cold-history candidates do not cover full target',
          'core_hypothesis':'necessary extra temporal evidence/shared-nuisance advantage not established in these cases; no universal refutation'},
        canonical_main_row=None,full_S43_retraining=False,GPU_used=False)
    write_json(OUT/'decision.json',decision)
    lines=['# S43 B-R 最小原型与核心命题验证（2026-10-27）','',
      '**决定：当前 B-R 原型不进入扩大训练；下一步转 A-R 的局部方向约束验证。A-R 尚未获验证，也未替换 S43。**','',
      '以下正式九列表仅引用历史主行，用于身份对齐；本轮是小规模有限候选机制任务，没有新的完整递推主行。',
      (OUT/'reference_main_table.md').read_text().strip(),'',
      '## 执行范围与三组对照','',
      '- 全CPU；31项单元检查通过，另11项数据核验与2段实际FIT事件流的1/10/25/50ms离线在线前缀逐项一致。',
      '- 每轴30个合成context，训练96/验证48/测试96；两轴共480样本。独立性按context，绝不按相邻窗口。',
      '- 现有数据112个原生窗口：FIT64、SELECT32、锁定后zgz报告16，覆盖14录制；同窗口的root/index任务不计成两次独立观测。SELECT四录制只有两人。',
      '- 原始last LNES、last/first/count/valid/五段count简单增强、B-R三臂；同24×32观测、历史、β/K、候选和时间水位。',
      '- 共同RBF核岭精确求解；图像CNN/证据MLP各400步、seed3407、同批次日程，参数与loss见结果JSON。没有用统一不适配接口作唯一裁决。',
      '- β/K沿原S43外供条件；历史由公共固定初态的两次冻结S43因果前向得到。当前GT只作监督/评分，候选最佳误差只作标记oracle下界。',
      '- 现有数据有v2e来源和伪标签，不冒称真实传感器实验；目标为cutoff前1ms，历史比epoch起点早1ms，所有臂相同held-state近似。','',
      '## 结果（机制指标，不进入正式跟踪主表）','']
    for scope,r in reports.items():
      split=sources[scope][1]; lines.append(f'**{scope}**')
      for reader in ['common','adapted']:
       m=[r[reader][a][split] for a in ['lnes','enhanced','br']]
       lines.append('- '+('共同核读出' if reader=='common' else '接口适配读出')+
             '；LNES / 简单增强 / B-R：正确率 '+ ' / '.join(f'{v["accuracy"]*100:.2f}%' for v in m)+
             '；候选端点绝对关节误差 '+ ' / '.join(f'{v["absolute_joint_mm"]:.3f}' for v in m)+' mm。')
      lines.append(f'- GT最佳候选下界 {r["common"]["br"][split]["oracle_candidate_mm"]:.3f} mm，属于评分诊断；不是模型输入。')
      lines.append('')
    lines += ['## 三个核心实验的归因','',
      '1. **时间证据（E1）**：细时间破坏保持LNES与简单增强逐项完全相同。合成root测试中B-R解析/共同读出正确率在破坏前后均100%，适配读出正确率也未变化；index的分类正确率同样未变。匹配生成器下的正收益因此不能归于本轮被破坏的必要时间信息，更可能包含物理归纳偏置/样本效率。该破坏实验也不是两条合法轨迹同码而目标不同的充分证明。',
      '2. **几何/前态（E2）**：一致重编号、纯数值换参考和同c0刷新对完整重放通过；错误证据身份会损坏选择，但这首先是通用cost-bank绑定。额外固定axis7→16控制的正确分支本身无可辨证据，未得到MANO几何必要性的阳性。0/5/15mm前态扰动无一致恢复；root 5mm组刷新变差、15mm部分纠回，index受覆盖/可见性限制。',
      '3. **共享不确定性/生命周期（E3）**：相同核与phase下，仅改变η混合账；本轮任务未显示共享处理的明确风险优势。全同路径/全遮挡负对照保持均匀概率；512小epoch有界，真实source保重数。逐bin prior-remix是故意不一致近似，不是另一套相机模型。',
      '', '## 唯一一轮科学修正','',
      '第一轮已有数据中，解析证据高度自信却无稳定任务收益；共同/适配读出已拟合训练，优先归因成像假设/覆盖与校准。新增全窗极性解释分支，保持几何/观测/候选不变，仅以FIT拟合映射prior和证据温度。',
      f'FIT最终选 flip_prior={calibration["selected"]["flip_prior"]:g}、T={calibration["selected"]["temperature"]:g}。128训练任务行来自同64窗口×两轴，不是128独立观测。',
      '选择零翻转权重不认证物理ON/OFF；温度是nearest-candidate有监督分数校准。它不改argmax，且被学习读出标准化抵消；因此改善NLL不能写成恢复运动信息或修好真实后验。',
      'SELECT root解析NLL由 '+f'{load("results/existing_root_x_selection_v1c.json")["analytic"]["analytic"]["validation"]["nll"]:.3f} 降至 {reports["existing_SELECT/root_x"]["analytic"]["analytic"]["validation"]["nll"]:.3f}'+
      '，任务风险无相应改善；index同样主要减轻过度自信。没有加步数、换seed或依据zgz再修。',
      '', '## 成本与实现局限','',
      '- B-R包含实际MANO三角面渲染、完整bin phase推进和owned query快照；不能只报小证据向量或query核。',
      '- 完整阶段耗时和数组/source驻留见 [benchmark.json](benchmark.json)；synthetic缓存的render_seconds=0仅表示复用缓存，不代表几何免费。',
      '- 修正产物中的query_seconds继承M3旧参考，不作为M6最终成本；独立benchmark单测M3/M6真实owned query。当前M6实现即使权重选0也仍计算双解释，未假称已经剪枝。',
      '- 冻结S43历史warmup为三臂共同成本；计时仅CPU回放，没有传感器到达时钟或完整服务队列，因此不声称sensor-to-mesh时延。',
      '- 本轮没有跑完整闭环、原W50/S39栈正式主行或全51D训练，正式S43/S39不变。',
      '', '## 最终判断与可复核入口','',
      'SELECT四项共同/适配×root/index的预注册继续门均未过。B-R匹配合成root的正控证明原型可工作，但时间必要性、关节稳健性、现有数据收益及成本价值没有同时闭合。当前实现停止扩大训练，转入A-R局部方向/可观测性假说；不宣布A-R已有效，也不将当前失配写成“原始事件没有更多信息”。',
      '', '- [机器决定](decision.json)、[CSV原始机制汇总](mechanism_results.csv)、[完整汇总JSON](results_summary.json)。',
      '- [契约日志](all_tests_final.log)、[实际数据前缀核验](existing_contracts.json)、[E2/E3诊断](diagnostics.json)、[错误自由度控制](diagnostics_wrong_dof.json)。',
      '- [独立审查](independent_audit.json)、[校准](calibration_v2.json)、[最终锁定](final_lock.json)、[读zgz前决定](pre_zgz_decision.json)。',
      '- 代码与命令：[README](../../tools/s43_br/README.md)。','',
      '工程历史保留：极小overflow正尾的浮点相减缺陷、公共条件显式补齐和固定cold-start对象8B差异的审计修订均在相关sidecar；旧v0及不公平的selection_v1结果不参与裁决。']
    report=OUT/'report.md'
    if report.exists(): raise FileExistsError(report)
    report.write_text('\n'.join(lines)+'\n')
    write_json(OUT/'frozen_sources_integrity.json',integrity)
    print(report)


if __name__=='__main__': main()
