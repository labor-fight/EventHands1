import numpy as np
from tools.s43_revisit.witness_null_diagnosis import threshold_native,difference


def test_static_radiance_is_genuine_zero_event_null():
    events=threshold_native(np.full((101,2,2),.15))
    assert events.shape==(0,4)


def test_multiset_difference_counts_true_repeats_and_ignores_same_ms_order():
    a=np.array([[1,2,0,0]]*4+[[3,4,1,1]],np.int64)
    assert difference(a,a[::-1])['exact_collision']
    b=np.array([[1,2,0,0]]*2+[[3,4,1,1]],np.int64)
    d=difference(a,b);assert d['L1_count']==2 and d['L2_count']==2 and d['differing_native_coordinates']==1


def test_shared_phase_threshold_repeat_is_deterministic_without_noise():
    images=np.full((3,2,2),.15);images[1]=.70
    a=threshold_native(images);b=threshold_native(images)
    np.testing.assert_array_equal(a,b)
    assert np.all(a[a[:,2]==0,3]==1) and np.all(a[a[:,2]==1,3]==0)
