"""Check provenance and local document links without rerunning task scores."""
from pathlib import Path
import hashlib
import json
import re
import urllib.parse

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/s43_revisit_20261028'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


MERGED_DOCS=ROOT/'docs/网络结构与可观测性诊断研究.md'
DOC_ARCHIVE=ROOT/'outputs/s43_docs_merge_20261028/originals'


def locate(name):
    """2026-10-28: four S43 documents were merged into the two maintained documents and deleted;
    a frozen dependency that named one of them is checked against its byte-identical archived copy."""
    path=ROOT/name
    if path.exists() or not name.startswith('docs/S43'):return path,False
    return DOC_ARCHIVE/Path(name).name,True


def main():
    protocol=json.loads((OUT/'stage1_protocol.json').read_text())
    located={p:locate(p) for p in protocol['dependencies']}
    dependency={p:located[p][0].exists() and sha(located[p][0])==expected for p,expected in protocol['dependencies'].items()}
    archived={p:str(located[p][0].relative_to(ROOT)) for p in protocol['dependencies'] if located[p][1]}
    if not all(dependency.values()):raise AssertionError({p:v for p,v in dependency.items() if not v})
    battle=json.loads((OUT/'battle/summary.json').read_text())
    assert battle['R1_cells']==battle['R2_cross_reviews']==100
    assert len(battle['normalized_cells'])==100 and not battle['selection_by_vote']
    stage2=json.loads((OUT/'stage2_protocol.json').read_text())
    assert sha(OUT/'selected_prototype_spec.md')==stage2['design_sha256']
    assert sha(OUT/'battle/summary.json')==stage2['battle_sha256']
    # Later artifacts are optional while a phase is running. Check identities
    # without importing model code, opening target arrays or recomputing scores.
    stages={}
    actual=OUT/'field_query_engineering/actual_sourcefreeze.json'
    if actual.exists():
        freeze=json.loads(actual.read_text())
        failures=[];identities=0
        for group in ('dependencies','sources','engineering'):
            for name,expected in freeze[group].items():
                file=Path(name)
                if not file.is_absolute():file=ROOT/file
                identities+=1
                if not file.exists() or sha(file)!=expected:failures.append(str(file))
        if failures:raise AssertionError({'Stage3_frozen_identity_changed':failures})
        stages['A_actual_sourcefreeze']=dict(sha256=sha(actual),identities_checked=identities,unchanged=True)
        completion=OUT/'field_atomic_v1/queue_complete.json'
        stages['A_queue_complete']=completion.exists()
        # Engineering evidence is separate from a task/training completion gate.
        combined=OUT/'field_query_engineering/root_combined_tests.json'
        if combined.exists():
            check=json.loads(combined.read_text())
            assert check['status']=='PASS'
            assert all(sha(ROOT/name)==identity for name,identity in check['source_identities'].items())
            stages['new_combined_engineering_tests']=check['output']
    checked=[];missing=[]
    # 2026-10-28: the three S43 documents now live in the research document (附录 N/O/P) with the
    # same block markers; the logical names below are kept for the receipt.
    blocks={'S43表征调研.md':('<!-- S43_REVISIT_STAGE1_BEGIN -->','<!-- S43_REVISIT_STAGE1_END -->'),
            'S43_三种表征方案与主方案定义_第二轮重设计_20261027.md':('<!-- S43_REVISIT_DESIGN_BEGIN -->','<!-- S43_REVISIT_DESIGN_END -->'),
            'S43_100单元交叉审查_第二轮重设计_20261027.md':('<!-- S43_REVISIT_BATTLE_BEGIN -->','<!-- S43_REVISIT_BATTLE_END -->')}
    for name in (*blocks,'EXPERIMENT_RECORDS.md'):
        path=ROOT/'docs'/name if name=='EXPERIMENT_RECORDS.md' else MERGED_DOCS;text=path.read_text()
        # Only the new current-round blocks: old historical paths are not new links.
        if name=='EXPERIMENT_RECORDS.md':text=text[text.index('<a id="s43-revisit-prereg-20261028">'):]
        else:
            start,end=blocks[name];text=text[text.index(start):text.index(end)]
        for target in re.findall(r'\]\(([^\n)]+)\)',text):
            target=target.strip('<>')
            if re.match(r'^[a-z]+://',target):continue
            raw,_,fragment=target.partition('#')
            local=(path.parent/urllib.parse.unquote(raw)).resolve() if raw else path
            exists=local.exists();checked.append(dict(document=name,target=target,exists=exists))
            if not exists:missing.append(checked[-1])
            if exists and fragment.startswith('s43-') and local.suffix=='.md':
                anchor=f'id="{fragment}"'
                if anchor not in local.read_text():missing.append(dict(document=name,target=target,missing_explicit_anchor=True))
    if missing:raise AssertionError(missing)
    result=dict(schema='s43_revisit_record_validation_v1',status='PASS',
                old_frozen_dependencies_match=dependency,dependencies_checked_from_docs_merge_archive=archived,
                current_local_links=len(checked),
                missing_links=missing,battle_cells=100,stage2_design_and_battle_sha_match=True,
                later_stage_evidence=stages,
                task_training_or_scoring_executed=False,
                note='Validation reads saved artifacts and documents only; it does not count as a new task experiment.')
    (OUT/'record_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',dependencies=len(dependency),links=len(checked),battle_cells=100)))


if __name__=='__main__':main()
