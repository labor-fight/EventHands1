"""Generate integration diagnostics from complete traces, not canonical rows."""
from pathlib import Path
import csv,datetime,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/s43_integration_20261027'
ARMS=['a0_frozen','a1_legacy','a2_enhanced','a3_br','a4_br_adapter','a5_br_readout']
SEQS=['ycy_local','ycy_global','zgz_local','zgz_global']
LABELS=['A0 原 S43（冻结参照）','A1 LNES + conv1 小训练','A2 简单增强 + 适配/conv1',
        'A3 B-R + 适配/conv1','A4 B-R 仅适配器','A5 B-R + 适配/conv1/原读出 *']


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def main():
    data={};rows=[]
    for arm in ARMS:
        data[arm]={}
        for seq in SEQS:
            path=OUT/'replay'/f'{arm}_{seq}_public.json';r=read(path)
            assert r['full_recording'] and r['all_valid_native_events_consumed']
            assert r['runs_processed']==r['valid_runs_total'] and r['actual_native_events']==r['expected_native_events']
            assert r['actual_ordered_source_digest']==r['expected_ordered_source_digest']
            assert all(r['source_files_unchanged'].values()) and sha(r['npz'])==r['npz_sha256']
            data[arm][seq]=r
            rows.append({'arm':arm,'sequence':seq,'n':r['scored_packets'],**r['metrics'],
                'source_events':r['actual_native_events'],'unsupported_holds':r['unsupported_hold_packets']})
    with (OUT/'stream_results.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    initial=read(OUT/'smoke.json')['results'][0]['params']['total']
    train={a:read(OUT/'train'/a/'train.json') for a in ARMS[1:]}
    header='| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE(递推) | Latency | FLOPs/step | Params |'
    table=[header,'|---|---|---|---|---|---|---|---|---|']
    aggregates={}
    for arm,label in zip(ARMS,LABELS):
        local=data[arm]['zgz_local'];glob=data[arm]['zgz_global'];pair=[local,glob]
        count=sum(r['scored_packets'] for r in pair)
        overall=sum(r['metrics']['mpjpe_ra_mm']*r['scored_packets'] for r in pair)/count
        seconds=sum(sum(r['timing'][key] for key in ['ingest_wall_seconds','query_wall_seconds','commit_wall_seconds'])+
                    sum(run.get('initial_reset_wall_seconds',0) for run in r['runs']) for r in pair)
        params=initial if arm=='a0_frozen' else train[arm]['params']['total']
        table.append(f'| {label} | {local["metrics"]["mpjpe_ra_mm"]:.2f} | {glob["metrics"]["mpjpe_ra_mm"]:.2f} '
          f'| {local["metrics"]["mpvpe_ra_mm"]:.2f} | {glob["metrics"]["mpvpe_ra_mm"]:.2f} | {overall:.2f} '
          f'| {seconds/count*1000:.2f} ms CPU | — | {params/1e6:.3f} M |')
        aggregates[arm]={'zgz_public_RA':overall,'main_cpu_ms':seconds/count*1000,'params':params,
              'complete_recordings':4,'scored_packets':sum(r['scored_packets'] for r in data[arm].values()),
              'native_events':sum(r['actual_native_events'] for r in data[arm].values()),
              'unsupported_holds':sum(r['unsupported_hold_packets'] for r in data[arm].values()),
              'duplicates_checked':sum(len(r['duplicate_query_checks']) for r in data[arm].values()),
              'peak_accounted_bytes':max(r['peak_accounted_memory']['total_accounted_bytes'] for r in data[arm].values())}
    (OUT/'stream_table.md').write_text('\n'.join(table)+'\n')
    write(OUT/'summary.json',{'scope':'public initialization + canonical 2-window warmup; W50 raw; diagnostic, not canonical main-row',
                            'aggregates':aggregates,'training':{a:{k:r[k] for k in ['initial_fit_loss','final_fit_loss','initial_SELECT_loss','final_SELECT_loss','steps','training_seconds','params']} for a,r in train.items()},
                            'raw_metrics':rows})
    protocol=read(OUT/'protocol.json')
    unchanged={p:sha(ROOT/p)==value for p,value in protocol['production_sources'].items()}
    assert all(unchanged.values())
    verify=read(OUT/'legacy_reproduction.json');short=read(OUT/'legacy_reproduction_short_runs.json')
    assert verify['PASS'] and short['PASS']
    old_trace=OUT/'replay_attempt1/a0_frozen_ycy_local_public.npz'
    new_trace=OUT/'replay/a0_frozen_ycy_local_public.npz'
    with np.load(old_trace) as a,np.load(new_trace) as b:
        exact_old=all(np.array_equal(a[k],b[k]) for k in ['pred','gt','end_ms','run_id'])
    assert exact_old
    hold_reasons={}
    for arm in ARMS:
        for seq,r in data[arm].items():
            if r['unsupported_hold_packets']:
                with np.load(r['npz']) as z:hold_reasons[f'{arm}/{seq}']=np.unique(z['status'][z['unsupported_hold']]).tolist()
    decision={'status':'COMPLETE_ENGINEERING_NEGATIVE_EFFECT','default_branch':'legacy',
          'BR_enabled_by_default':False,'promote_BR':False,'architecture_and_MANO_loss_preserved':True,
          'original_sources_unchanged':unchanged,'public_complete_replays':24,
          'public_scored_packets':sum(v['scored_packets'] for v in aggregates.values()),
          'public_consumed_native_events':sum(v['native_events'] for v in aggregates.values()),
          'legacy_standard_predictions_exact':verify['predictions']+short['predictions'],
          'legacy_old_public_trace_exact_after_bound_fix':exact_old,
          'increments':'A1 input-conv adaptation; A2 enhancement; A3 B-R; A4 adapter-only; A5 posthoc original-readout adaptation',
          'readout_ablation_is_posthoc':True,'unsupported_reasons':hold_reasons,
          'limits':['100-step, one seed, 2-step TBPTT=1 rather than full training',
                    'new representation uses a frozen LNES-pretrained trunk; A5 releases only old final readout',
                    'fixed root-x hypothesis set/constant uncalibrated radiance, no full articulation coverage',
                    'uniform-phase conditional restart at arbitrary continuous state commit',
                    'near-pi chart can force explicit hold; not silent pose clipping',
                    'CPU masked-dense execution, not sparse/GPU service benchmark',
                    'S39 W200/filter stack not validated; public and GT-init protocols separated']}
    write(OUT/'decision.json',decision)
    lines=['# S43 B-R 接入与流式验证（2026-10-27）','',
      '**接入、训练和完整回放已经完成；效果不支持替换原 S43。默认仍为 legacy，B-R 仅通过显式配置开启。**','',
      '## 本轮完整递推诊断表','',
      '下表来自锁定参数后的完整 zgz_local/global 公共初态流：每有效段公共初态、两窗原canonical S43预热，W50/raw，seed3407；不是历史正式主行。Latency为CPU主段耗时摊销，含原生ingest/表示/查询/commit，含重复查询审计的开销；不含共同预热、尾段和GT评分，这些在逐录制JSON另列。FLOPs未测，不能与旧GPU主行的Latency直接比较。','',
      '\n'.join(table),'',
      '* A5为看到初轮长递推退化后追加的读出训练能力消融，不能当预注册独立泛化证据。模型总结构参数不变，但释放了26,163个既有读出参数。','',
      '## 实现与旧分支复现','',
      '- 只新增tools/s43_integration、configs/s43_integration和测试。原model/model.py、MANO、事件构造、训练/评价入口及原型核心的冻结SHA全部保持。',
      '- 优先替换事件表示及必要输入适配，保留原CNN、prev-render/MLP、51D输出、MANO解码和原SO3/trans/FK监督。A5仅放开已有rn.fc训练，单独标记；没有新增主干/损失。',
      '- 旧配置默认legacy；new wrapper显式消费自己的四个新配置键，原模型其余键仍严格校验。新checkpoint带schema/mode/temperature/K，序列状态不混入权重。',
      f'- 两条完整旧标准协议诊断（ycy_global与含49个短段的zgz_local）共{verify["predictions"]+short["predictions"]}步，新旧pred/GT/end/run逐数组完全相同，最大预测差0、四项指标差0。短段先跳过随机噪声的边界已修。原两录制共享RNG正式栈仍由旧入口复现，不把单序列CLI冒充整套主行。',
      '- 原A0公共ycy_local在修正内存上限前后的完整预测也逐数组完全相同。','',
      '## 训练与在线同义性','',
      '- reset/ingest/readout/release共用核心；训练batch按同一函数执行。两步clip第二步由本臂预测的detach手态重建，TBPTT=1，GT不进入历史/ROI/模板。缓存lead仅复用不可变canonical warmup条件。',
      '- query只输出；commit才改变手态。连续头输出候选外状态后明确uniform-phase conditional restart，不假称精确Bayes phase交接。',
      '- 默认每epoch最多262144个source、最多16序列。padding行不读事件/历史，适配器跳过，监督在原loss前过滤；NaN padding在真实checkpoint smoke中过滤通过。',
      '- 同时验证了混序列/重启隔离、相同时间组、分块一致、空流、UID幂等/真实重数、future staging不泄漏、只读query和closed epoch不重复提交。','',
      '## 实际运行的证据','',
      '- 38项单元契约全过（明确包含fake provider/tiny模型，不冒充真实模型）。另有真实S43 forward/backward smoke、六组真实MANO等价渲染、原生8个prefix逐值parity、200个实际BR prefix在线/batch对照。',
      '- 真实ch_local/lfz_global混序列交错100个ms包，四种表示数组逐值相同；batch与独立预测误差约3e-7，单序列reset不改变另一个，结束全部release。',
      '- 四个初始训练臂各100次实际optimizer更新，索引/clip初态配对；A5同100次、同样本索引。原主干/BN/物理解码张量不变；A5明确例外是原conv1/rn.fc权重。',
      '- FIT loss仅固定前8 clip，SELECT loss全部16；下降是短clip拟合，不等于长流稳定。32 FIT clip和16 SELECT clip均来自真实相邻原生事件窗口，输入不带target。','']
    for a,r in train.items():
        lines.append(f'- {a}：FIT {r["initial_fit_loss"]:.6f} → {r["final_fit_loss"]:.6f}；SELECT {r["initial_SELECT_loss"]:.6f} → {r["final_SELECT_loss"]:.6f}；trainable {r["params"]["trainable"]}。')
    lines+=['',f'- 共24条完整录制级回放、{decision["public_scored_packets"]}个评分packet、{decision["public_consumed_native_events"]:,}次原生source消费（同四录制按六模型重复，不是独立样本数）。所有valid run、预热和短尾已消费，计数和顺序摘要完全一致，源SHA在各次执行内不变。valid-runs外无标签的事件未评分，另列数量。',
       '- B-R A3/A4无unsupported；A5存在的近πchart hold见decision.json/逐packet trace，hold仍消费事件并计入全部评分，不裁段、不用原LNES旁路救结果。','',
       '## 前态误差与增量成本','',
       '- 固定SELECT四窗口，对0/5/15mm root-x前态偏移，legacy/enhanced事件图严格不变，但原prev-render变化；B-R证据/模板随几何条件变化。fresh相对stale的绝对/RA变化不一致，15mm下RA可改善而绝对误差更差，不能宣称普遍纠错。',
       '- 真正局部的是新source/触及槽写入；完整ms phase推进全KMPQ，重置后的MANO/FK/三角面/模板仍全局，原生图读出全图归约，CNN仍完整前向。',
       '- C++常色三角覆盖保留全部51个时间点，与原provider的状态/FK/整数光度逐项一致；没有通过时间降采样偷减工作。',
       '- 当前执行器是原forward的masked-dense CPU调试路径，尚未接部署SparseExecutor/CUDA graph。在线表示的驻留状态有界；导出轨迹/审计日志O(T)另算，统计内存不是完整进程RSS。','',
       '## 失败现场与归因边界','',
       '- 首次批回放触发10万source界限及并发统计代码的来源保护，失败/中断现场在replay_attempt1*。上限依据原始offset元数据改为262144（实测50ms峰值最高157293）；未改每事件数值、权重或评价门。随后冻结依赖完整重跑，不能把第一次失败当完成。',
       '- 效果失败不归结为表示实现不一致：真实前缀parity和旧分支复现均通过。旧LN预训练主干对新输入存在适配限制，A4/A5已分开测试；A5若改善只能说明读出训练能力的一部分作用。',
       '- 当前K3只比较root-x路径，常色光度和phase重启未经真实相机标定，可能丢掉关节事件证据。短clip训练与长闭环条件分布不同，也未排除完整训练的潜力；不据此证明原始事件没有额外信息。','',
       '## 交付与剩余问题','',
       '- 代码/接口/命令：[README](../../tools/s43_integration/README.md)；[机器判断](decision.json)；[逐录制CSV](stream_results.csv)；[汇总JSON](summary.json)。',
       '- 实测入口：[smoke](smoke.json)、[padding](padding_smoke.json)、[混序列](mixed_sequence_smoke.json)、[前态扰动](perturb.json)、[旧标准复现](legacy_reproduction.json)、[短段复现](legacy_reproduction_short_runs.json)。',
       '- 余项：近π需要稳健chart重基而非hold；有限候选/光度域差与phase重启；更长自回馈训练和完整预注册训练；真正稀疏/GPU服务执行、隔离时延与完整FLOPs；S39 W200/filter栈。没有授权或声称已完成6000步/多seed/真实传感器泛化。',
       '- 工程分支保留供后续研究，默认不启用；没有将本轮权重写入旧S43/S39指针或正式主表。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(OUT/'report.md')


if __name__=='__main__':main()
