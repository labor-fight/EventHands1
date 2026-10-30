"""Cross-interface controls for the scientific comparisons, not head accuracy."""
import unittest
import numpy as np
from tools.s43_br.representations import pixel_features,temporal_intervention,SENSOR
from tools.s43_br.correction import dual_evidence
from tools.s43_br.carrier import encode_batch


class ExperimentContracts(unittest.TestCase):
    def test_time_intervention_preserves_both_baselines_but_changes_raw(self):
        rng=np.random.default_rng(4)
        counts=rng.integers(0,4,(50,7,2))
        altered=temporal_intervention(counts,8)
        self.assertTrue(np.any(counts!=altered))
        for before,after in zip(pixel_features(counts),pixel_features(altered)):
            np.testing.assert_array_equal(before,after)

    def test_dual_paths_equal_exact_stored_polarity_reinterpretation(self):
        rng=np.random.default_rng(7)
        paths=np.cumsum(rng.integers(-3,4,(3,3,4,8)),axis=-1)
        counts=rng.integers(0,5,(7,4,2))
        dual,_,_,_=dual_evidence(paths,counts)
        positive=encode_batch(paths,counts,**SENSOR)['log_likelihood']
        flipped=encode_batch(paths,counts[...,::-1].copy(),**SENSOR)['log_likelihood']
        np.testing.assert_allclose(dual[:,0],positive,atol=1e-10,rtol=0)
        np.testing.assert_allclose(dual[:,1],flipped,atol=1e-10,rtol=0)


if __name__=='__main__': unittest.main()
