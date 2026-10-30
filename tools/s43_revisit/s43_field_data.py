"""Target-free chronological lanes and separately gated lazy supervision.

Only MeanShapeInputReader is constructed by the input schedule. No input
record contains LNES, auxiliary shape, GT pose, or previous GT. Source cursor
advancement is explicit and follows successful runtime commit.
"""
from __future__ import annotations
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.s43_br.data import DATA_ROOT, FIT7, PUBLIC_INITIAL_STATE, recording_dirs
from tools.s43_rel.mean_shape_replay import MeanShapeInputReader


@dataclass(frozen=True)
class Run:
    recording: str
    run_id: int
    start_ms: int
    stop_ms: int


@dataclass(frozen=True)
class FieldInputBatch:
    items: tuple
    releases: tuple
    batch_id: int
    token: str

    def __len__(self):return len(self.items)
    def __iter__(self):return iter(self.items)
    def __getitem__(self,index):return self.items[index]


def make_input_batch(items, *, releases=(), batch_id=0):
    """Public engineering/online adapter for the same runtime input contract."""
    items=tuple(items);releases=tuple(releases);h=hashlib.sha256(str(int(batch_id)).encode())
    for kind, rows in (('items',items),('releases',releases)):
        for row in rows:
            h.update(kind.encode())
            if not row.get('valid',True) or row.get('padding',False):
                h.update(repr(('padding',row.get('lane'))).encode());continue
            for key in ('seq_id','recording','start_ms','cutoff_ms','packet_ordinal','source_range','reset','valid','padding'):
                h.update(repr(row.get(key)).encode())
            if row.get('valid',True) and row.get('events') is not None:
                h.update(np.asarray(row['events']).tobytes())
            for key in ('initial_state','beta_input','K_source'):
                if key in row:h.update(key.encode());h.update(np.asarray(row[key]).tobytes())
    return FieldInputBatch(items,releases,int(batch_id),h.hexdigest())


class TargetFreeRunLanes:
    """Read-only valid-run lanes, W50/hop50, peek/commit cursor transaction.

    Run ordering and packet timestamp reconstruction follow the old continuous
    lane algorithm. The reader and returned fields are deliberately different:
    only events, K, public beta0/initial pose and scheduling metadata.
    """
    def __init__(self,root=DATA_ROOT,*,subjects=FIT7,recordings=None,batch_size=8,
                 seed=3407,finite=False,runs=None,reader_factory=MeanShapeInputReader):
        self.root=Path(root);self.subjects=tuple(subjects);self.batch_size=int(batch_size)
        if self.batch_size<1 or 'zgz' in self.subjects:
            raise ValueError('positive lanes; no report-subject input scope in training adapter')
        self.finite=bool(finite);self.readers={}
        names=sorted(recordings if recordings is not None else recording_dirs(self.root))
        for name in names:
            if name.split('_')[0] not in self.subjects:continue
            self.readers[name]=reader_factory(name,self.root)
        self.runs=[]
        if runs is None:
            for name,reader in self.readers.items():
                self.runs.extend(Run(name,i,int(a),int(b)) for i,(a,b) in enumerate(reader.runs) if int(b)-int(a)>=50)
        else:
            for run in runs:
                run=Run(**run) if isinstance(run,dict) else run
                reader=self.readers[run.recording]
                if not any(run.start_ms==int(a) and run.stop_ms<=int(b) for a,b in reader.runs):
                    raise ValueError('engineering run must begin at real valid boundary and lie inside its run')
                if run.stop_ms-run.start_ms<50:raise ValueError('query run shorter than50ms')
                self.runs.append(run)
        if not self.runs:raise ValueError('no eligible target-free runs')
        rng=np.random.default_rng(int(seed))
        self._state=dict(rng=rng.bit_generator.state,order=np.empty(0,np.int64),position=0,traversal=0,
            lanes=[None]*self.batch_size,batch_id=0,outputs=0,events=0,resets=0,tail_events=0,tail_ms=0)
        self._pending=None

    def _new_run(self,state,lane):
        if state['position']>=len(state['order']):
            if self.finite and state['traversal']>0:return None
            rng=np.random.default_rng();rng.bit_generator.state=state['rng']
            state['order']=rng.permutation(len(self.runs));state['rng']=rng.bit_generator.state
            state['position']=0;state['traversal']+=1
        run=self.runs[int(state['order'][state['position']])];state['position']+=1;state['resets']+=1
        return dict(run=run,cursor_ms=run.start_ms,ordinal=0,
            seq_id=f"lane{lane}:pass{state['traversal']}:{state['position']}:{run.recording}:{run.run_id}")

    @staticmethod
    def _packet(reader,start,stop):
        packet=reader.packet(start,stop)
        packet['events'].flags.writeable=False
        return packet

    def peek_batch(self):
        if self._pending is not None:return self._pending[0]
        state=copy.deepcopy(self._state);items=[];releases=[]
        for lane in range(self.batch_size):
            current=state['lanes'][lane]
            reset=current is None or current['cursor_ms']+50>current['run'].stop_ms
            if reset and current is not None:
                run=current['run'];start=current['cursor_ms'];stop=run.stop_ms
                release=dict(seq_id=current['seq_id'],recording=run.recording,valid=True,
                    start_ms=start,cutoff_ms=stop,packet_ordinal=current['ordinal'],events=None)
                if start<stop:
                    release.update(self._packet(self.readers[run.recording],start,stop))
                    state['events']+=len(release['events']);state['tail_events']+=len(release['events']);state['tail_ms']+=stop-start
                releases.append(release)
            if reset:current=state['lanes'][lane]=self._new_run(state,lane)
            if current is None:
                items.append(dict(lane=lane,valid=False,padding=True));continue
            run=current['run'];start=current['cursor_ms'];stop=start+50
            reader=self.readers[run.recording];packet=self._packet(reader,start,stop)
            item=dict(lane=lane,valid=True,padding=False,seq_id=current['seq_id'],reset=reset,
                recording=run.recording,run_id=run.run_id,run_start_ms=run.start_ms,run_stop_ms=run.stop_ms,
                start_ms=start,cutoff_ms=stop,target_ms=stop-1,packet_ordinal=current['ordinal'],
                initial_state=PUBLIC_INITIAL_STATE.copy(),beta_input=np.zeros(10,np.float32),K_source=reader.K_cam.copy(),**packet)
            if set(item)&{'lnes','betas','beta_target','target','targets','previous_GT'}:
                raise AssertionError('forbidden data entered input item')
            items.append(item);current['cursor_ms']=stop;current['ordinal']+=1
            state['outputs']+=1;state['events']+=len(packet['events'])
        batch=make_input_batch(items,releases=releases,batch_id=state['batch_id'])
        state['batch_id']+=1;self._pending=(batch,state)
        return batch

    next_batch=peek_batch

    def commit_batch(self,batch):
        if self._pending is None or batch is not self._pending[0]:raise ValueError('foreign or duplicate cursor commit')
        self._state=self._pending[1];self._pending=None

    @property
    def exhausted(self):
        return self.finite and self._state['traversal']>0 and self._state['position']>=len(self.runs) and all(x is None for x in self._state['lanes'])

    def _identity(self):
        return dict(root=str(self.root),subjects=self.subjects,batch_size=self.batch_size,finite=self.finite,
                    runs=[asdict(r) for r in self.runs],recordings=sorted(self.readers))

    def state_dict(self):
        if self._pending is not None:raise RuntimeError('checkpoint cursor after completed runtime/cursor commit, not an unacknowledged peek')
        return copy.deepcopy(dict(schema='s43_field_lane_schedule_v1',identity=self._identity(),state=self._state))

    def load_state_dict(self,state):
        if state['schema']!='s43_field_lane_schedule_v1' or state['identity']!=self._identity():raise ValueError('sampling checkpoint schedule differs')
        self._state=copy.deepcopy(state['state']);self._pending=None

    def manifest(self):
        return dict(self._identity(),window_ms=50,hop_ms=50,warmup='runtime first2own outputs, not supervision',
            initial_state=PUBLIC_INITIAL_STATE.tolist(),beta_input='zeros10',GT_input=False,LNES_constructed=False,
            cursor='peek/commit; checkpoint only after acknowledged complete window',
            short_tail='batch.releases consumes all valid tail once; no query/feedback/loss',
            padding='invalid item has no events/state/feedback; releases are separate real-source operations')


@dataclass(frozen=True)
class SupervisionBatch:
    target51: np.ndarray
    beta_target: np.ndarray
    mask: np.ndarray
    input_token: str


class LazySupervisionReader:
    """Open supervision only after the runtime forward has built all inputs.

    Warmup/padding targets are not read. No loaded shape/pose is returned to
    input readers or runtime state. A poisoned factory can test this boundary.
    """
    def __init__(self,root=DATA_ROOT,*,reader_factory=None):
        self.root=Path(root);self._factory=reader_factory;self._readers={};self.reads=0

    def read(self,result):
        if not getattr(result,'inputs_complete',False) or not getattr(result,'input_token',None):
            raise RuntimeError('all runtime inputs and geometry must precede supervision')
        items=result.items;mask=np.asarray(result.supervised_mask,bool).copy()
        targets=np.zeros((len(items),51),np.float32);beta=np.zeros((len(items),10),np.float32)
        for i,(item,selected) in enumerate(zip(items,mask)):
            if not selected:continue
            name=item['recording']
            if name not in self._readers:
                if self._factory is None:
                    from tools.s43_br.data import SequenceReader
                    self._readers[name]=SequenceReader(name,self.root)
                else:self._readers[name]=self._factory(name,self.root)
            reader=self._readers[name]
            targets[i]=reader.target(item['target_ms']);beta[i]=reader.betas;self.reads+=1
        return SupervisionBatch(targets,beta,mask,result.input_token)
