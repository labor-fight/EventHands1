"""Resource scheduler contract tests with stubbed storage and isolated process cleanup.

Only the dedicated child process uses PR_SET_CHILD_SUBREAPER; pytest itself never does.
The final installed SOURCE is the actual repository tools/xag/run.py.
"""
import concurrent.futures
import contextlib
import ctypes
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock,patch
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'tools/xag/run.py'


def module(root):
 prep=types.ModuleType('tools.xag.prepare');prep.OUT=root;prep.CELLS=('a0f0',)
 prep.save=lambda p,v:p.write_text(json.dumps(v));prep.sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();prep.now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();prep.verify=lambda:None
 lock=types.ModuleType('tools.xag.process_lock');lock.acquire=lambda p:None;lock.inherited_fds=lambda:();lock.assert_no_orphan_jobs=lambda p:None
 tools=types.ModuleType('tools');tools.__path__=[];xag=types.ModuleType('tools.xag');xag.__path__=[]
 m=types.ModuleType('candidate_under_test');m.__file__=str(SOURCE);before=list(sys.path)
 with patch.dict(sys.modules,{'tools':tools,'tools.xag':xag,'tools.xag.prepare':prep,'tools.xag.process_lock':lock}):
  exec(compile(SOURCE.read_text(),str(SOURCE),'exec'),m.__dict__)
 sys.path[:]=before;m.ROOT=root;m.OUT=root;m.HOST_OOM_BASELINE=21
 (root/'protocol.json').write_text(json.dumps({'runs':[],'resources':{'host_memory_guard':m.HOST_GUARD}}));(root/'frozen_sources.json').write_text('{}')
 m.audit_available=lambda p:None
 return m


def reading(gib=64,oom=21):return {'at_utc':'synthetic','monotonic':time.monotonic(),'available_bytes':gib*1024**3,'oom_kill':oom}
def members(pid,start=10):return {pid:{'starttime':start,'session':pid,'state':'S'}}


class GuardTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='xag_guard_test_');self.root=Path(self.tmp.name);self.m=module(self.root);self.events=[]
  self.m.journal=lambda job,**values:self.events.append((job,values))
 def tearDown(self):self.tmp.cleanup()
 def process(self,pid=101):
  p=Mock();p.pid=pid;p.poll.return_value=-15;return p
 def run_fake(self,samples,waits):
  m=self.m;p=self.process();p.wait.side_effect=waits
  with patch.object(m.subprocess,'Popen',return_value=p),patch.object(m,'group_members',side_effect=lambda pid:members(pid)),patch.object(m,'host_memory',side_effect=samples),patch.object(m,'terminate_owned',return_value={'actions':[],'remaining':{}}) as clean:
   try:m.run_command('one',['fake'],[0,1]);error=None
   except BaseException as exc:error=exc
   return error,p,clean.call_args_list
 def test_historical_oom21_allows_success(self):
  e,p,c=self.run_fake([reading(),reading()],[0]);self.assertIsNone(e);self.assertFalse(c);self.assertEqual(self.events[-1][1]['status'],'PROCESS_EXITED');self.assertFalse(self.m.STOP.is_set())
 def test_two_low_samples_abort_with_one_second_wait_between(self):
  e,p,c=self.run_fake([reading(8),reading(8)],[subprocess.TimeoutExpired('x',1)])
  self.assertIn('BELOW16',str(e));self.assertEqual(p.wait.call_args_list[0].kwargs['timeout'],1);self.assertEqual(len(c),1);self.assertEqual(self.events[-1][1]['status'],'FAILED')
 def test_single_low_then_recovered_does_not_abort(self):
  e,p,c=self.run_fake([reading(8),reading(64),reading(64)],[subprocess.TimeoutExpired('x',1),0]);self.assertIsNone(e);self.assertFalse(c)
 def test_new_oom_aborts_immediately(self):
  e,p,c=self.run_fake([reading(oom=22)],[]);self.assertIn('OOM_COUNTER',str(e));self.assertFalse(p.wait.called);self.assertEqual(len(c),1)
 def test_exit_interval_oom_is_not_missed(self):
  e,p,c=self.run_fake([reading(),reading(oom=22)],[0]);self.assertIn('AT_EXIT',str(e));self.assertEqual(len(c),1)
 def test_initial_identity_read_failure_recovers_for_cleanup(self):
  m=self.m;p=self.process()
  with patch.object(m.subprocess,'Popen',return_value=p),patch.object(m,'group_members',side_effect=[OSError('first read failed'),members(101)]),patch.object(m,'terminate_owned',return_value={'remaining':{}}) as clean:
   with self.assertRaisesRegex(OSError,'first read failed'):m.run_command('one',['fake'],[0,1])
  self.assertEqual(clean.call_args.args[-1],10);self.assertTrue(m.STOP.is_set());self.assertEqual(self.events[-1][1]['status'],'FAILED')
 def test_popen_failure_stops_queue_without_signalling(self):
  m=self.m
  with patch.object(m.subprocess,'Popen',side_effect=OSError('launch failed')),patch.object(m,'terminate_owned') as clean:
   with self.assertRaisesRegex(OSError,'launch failed'):m.run_command('one',['fake'],[0,1])
  self.assertTrue(m.STOP.is_set());self.assertFalse(clean.called)
 def test_stop_before_launch_does_not_spawn(self):
  m=self.m;m.request_stop('earlier failure')
  with patch.object(m.subprocess,'Popen') as p:
   with self.assertRaisesRegex(RuntimeError,'earlier failure'):m.run_command('one',['fake'],[0,1])
  self.assertFalse(p.called)
 def test_nonzero_job_stops_other_thread_and_each_cleans_only_own(self):
  m=self.m;ready=threading.Event();p1=self.process(101);p2=self.process(202);cleaned=[]
  def popen(cmd,**kw):
   if cmd==['bad']:return p1
   ready.set();return p2
  def wait1(**kw):
   self.assertTrue(ready.wait(2));return 7
  def wait2(**kw):
   time.sleep(.05);raise subprocess.TimeoutExpired('other',kw['timeout'])
  p1.wait.side_effect=wait1;p2.wait.side_effect=wait2
  def clean(proc,job,token,start):cleaned.append((job,proc.pid));return {'remaining':{}}
  with patch.object(m.subprocess,'Popen',side_effect=popen),patch.object(m,'host_memory',side_effect=lambda:reading()),patch.object(m,'group_members',side_effect=lambda pid:members(pid)),patch.object(m,'terminate_owned',side_effect=clean):
   with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    fs=[pool.submit(m.run_command,'bad',['bad'],[0,1]),pool.submit(m.run_command,'other',['other'],[2,3])]
    errors=[f.exception(timeout=3) for f in fs]
  self.assertTrue(all(errors));self.assertEqual(set(cleaned),{('bad',101),('other',202)});self.assertTrue(m.STOP.is_set())
 def test_sigterm_stops_dispatch_and_prevents_pair_audit_or_finalize(self):
  m=self.m;started=[];lock=threading.Lock();released=threading.Event();handlers={};done={}
  (self.root/'protocol.json').write_text(json.dumps({'runs':[{'name':str(i)} for i in range(12)],'resources':{'host_memory_guard':m.HOST_GUARD}}))
  def train(r,pair):
   with lock:started.append(r['name'])
   self.assertTrue(released.wait(3))
  def run():
   try:m.main()
   except BaseException as e:done['error']=e
  with patch.object(sys,'argv',['candidate','train']),patch.object(m.signal,'signal',side_effect=lambda sig,handler:handlers.update({sig:handler})),patch.object(m,'train_one',side_effect=train),patch.object(m,'host_memory',side_effect=lambda:reading()),patch.object(m,'audit_pairs') as audit,patch.object(m.subprocess,'run') as final:
   t=threading.Thread(target=run);t.start();deadline=time.monotonic()+3
   while len(started)<4 and time.monotonic()<deadline:time.sleep(.01)
   self.assertEqual(len(started),4);handlers[signal.SIGTERM](signal.SIGTERM,None);released.set();t.join(3)
   self.assertFalse(t.is_alive());self.assertFalse(audit.called);self.assertFalse(final.called)
  self.assertEqual(len(started),4);self.assertIn('scheduler_signal',str(done['error']))
 def test_journal_error_does_not_skip_cleanup(self):
  m=self.m;m.journal=Mock(side_effect=OSError('journal unavailable'));p=self.process()
  with patch.object(m.subprocess,'Popen',return_value=p),patch.object(m,'group_members',side_effect=lambda pid:members(pid)),patch.object(m,'terminate_owned',return_value={'remaining':{}}) as clean,contextlib.redirect_stderr(io.StringIO()):
   with self.assertRaisesRegex(OSError,'journal unavailable'):m.run_command('one',['fake'],[0,1])
  self.assertEqual(clean.call_count,1);self.assertTrue(m.STOP.is_set())
 def cleanup_fake(self,groups,environ):
  m=self.m;m.HOST_GUARD=dict(m.HOST_GUARD,termination_grace_seconds=0);p=self.process()
  with patch.object(m,'group_members',side_effect=groups),patch.object(m.Path,'read_bytes',lambda path:environ(path)),patch.object(m.os,'killpg') as signals:
   result=m.terminate_owned(p,'own','token',10);return result,[x.args for x in signals.call_args_list]
 def owned_env(self,path):return f'XAG_STUDY_ROOT={self.root}\0XAG_ACTIVE_JOB=own\0XAG_PROCESS_TOKEN=token\0'.encode()
 def test_reaped_leader_live_descendant_is_still_owned(self):
  result,calls=self.cleanup_fake(lambda pid:{102:{'starttime':11,'session':101,'state':'S'}},self.owned_env);self.assertEqual(calls,[(101,signal.SIGTERM),(101,signal.SIGKILL)])
 def test_reused_leader_foreign_token_and_session_refused(self):
  for member,env in [(members(101,11),self.owned_env),(members(101),lambda p:b'XAG_ACTIVE_JOB=formal\0'),({101:{'starttime':10,'session':999,'state':'S'}},self.owned_env)]:
   result,calls=self.cleanup_fake(lambda pid:member,env);self.assertEqual(calls,[]);self.assertTrue(any('cleanup_error' in x for x in result['actions']))
 def test_member_disappears_but_owned_peer_signalled(self):
  def env(path):
   if '/102/' in str(path):raise FileNotFoundError('gone')
   return self.owned_env(path)
  result,calls=self.cleanup_fake(lambda pid:{**members(101),102:{'starttime':11,'session':101,'state':'S'}},env);self.assertEqual(len(calls),2)
 def test_killpg_already_exited_is_recorded(self):
  m=self.m;m.HOST_GUARD=dict(m.HOST_GUARD,termination_grace_seconds=0)
  with patch.object(m,'group_members',side_effect=lambda pid:members(pid)),patch.object(m.Path,'read_bytes',lambda path:self.owned_env(path)),patch.object(m.os,'killpg',side_effect=ProcessLookupError('gone')):
   result=m.terminate_owned(self.process(),'own','token',10)
  self.assertEqual(sum(x.get('status')=='already_exited' for x in result['actions']),2)
 def test_real_reaped_leader_worker_killed_unrelated_group_untouched(self):
  result=subprocess.run([sys.executable,'-S',str(Path(__file__).resolve()),'--owned-group-child',str(SOURCE.resolve())],
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                        capture_output=True,text=True,timeout=15)
  self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
  self.assertIn('OK',result.stderr)
 def _exercise_real_reaped_leader_worker(self):
  m=self.m;m.HOST_GUARD=dict(m.HOST_GUARD,termination_grace_seconds=.3)
  # Become subreaper only inside this private test process so an intentionally
  # orphaned grandchild can be waitpid-reaped rather than leaving a zombie.
  libc=ctypes.CDLL(None,use_errno=True);old=ctypes.c_int();self.assertEqual(libc.prctl(37,ctypes.byref(old),0,0,0),0);self.assertEqual(libc.prctl(36,1,0,0,0),0)
  child_pid=None;leader=None;sentinel=None
  try:
   sentinel=subprocess.Popen([sys.executable,'-S','-c','import time; time.sleep(30)'],start_new_session=True)
   dest=self.root/'child.pid';code='''import os,signal,sys,time
pid=os.fork()
if pid:
 time.sleep(.3)
 os._exit(7)
signal.signal(signal.SIGTERM,signal.SIG_IGN)
with open(sys.argv[1],'w') as f:f.write(str(os.getpid()))
while True:time.sleep(.1)
'''
   env=dict(os.environ,XAG_STUDY_ROOT=str(self.root),XAG_ACTIVE_JOB='own',XAG_PROCESS_TOKEN='token',CUDA_VISIBLE_DEVICES='')
   leader=subprocess.Popen([sys.executable,'-S','-c',code,str(dest)],env=env,start_new_session=True)
   ident=m.group_members(leader.pid)[leader.pid]['starttime'];self.assertEqual(leader.wait(timeout=2),7)
   self.assertTrue(dest.exists());child_pid=int(dest.read_text());self.assertIn(child_pid,m.group_members(leader.pid))
   result=m.terminate_owned(leader,'own','token',ident);self.assertEqual(result['remaining'],{});self.assertIsNone(sentinel.poll());self.assertTrue(any(x.get('signal')=='SIGKILL' for x in result['actions']))
   waited,status=os.waitpid(child_pid,0);self.assertEqual(waited,child_pid);child_pid=None
  finally:
   if child_pid is not None:
    try:os.kill(child_pid,signal.SIGKILL)
    except ProcessLookupError:pass
    try:os.waitpid(child_pid,0)
    except ChildProcessError:pass
   if leader is not None:
    try:leader.wait(timeout=1)
    except subprocess.TimeoutExpired:leader.kill();leader.wait()
   if sentinel is not None:
    sentinel.terminate();sentinel.wait(timeout=2)
   libc.prctl(36,old.value,0,0,0)

if __name__=='__main__':
 if len(sys.argv)==3 and sys.argv[1]=='--owned-group-child':
  SOURCE=Path(sys.argv[2]).resolve()
  result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([GuardTests('_exercise_real_reaped_leader_worker')]))
  raise SystemExit(0 if result.wasSuccessful() else 1)
 unittest.main(verbosity=2)
