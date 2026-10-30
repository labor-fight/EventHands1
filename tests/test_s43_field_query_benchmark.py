"""Resource-tool bookkeeping only; no timed stream or task labels."""
import numpy as np
from tools.s43_revisit.field_query_benchmark import CASES,bounded_history,numeric_arrays


def test_frozen_sixteen_case_domain_and_extra_query_clock():
    assert len(CASES)==16 and len(set(CASES))==16
    assert ('anatomical',25,50) in CASES
    for mode in ('field_core','hand_box','static','generic_trajectory','anatomical'):
        assert all((mode,hop,hop) in CASES for hop in (25,50,100))


def test_history_has_one_past_bracket_and_stays_bounded():
    old=[]
    for T in range(0,500,25):
        old=bounded_history(old,T)+[(T,np.zeros(51))]
        assert len(old)<=4
    assert [t for t,_ in old]==[400,425,450,475]


def test_numeric_arrays_counts_alias_once_and_excludes_python():
    x=np.zeros(51,np.float64)
    assert numeric_arrays({'a':x,'b':[x],'other':1})==408
