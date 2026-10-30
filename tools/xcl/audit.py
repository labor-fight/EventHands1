"""Native opt-in XCL provenance, independent of retired XAG launcher identity."""
import pytorch_lightning as pl
from tools.xcl.common import BASELINE, SOURCE_MANIFEST, read, save, sha

class XCLIdentityAudit(pl.Callback):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_fit_start(self, trainer, model):
        if trainer.global_rank == 0:
            save(self.output_dir / "xcl_binding.json", {
                "cell": model.cfg["XCL"]["CELL"], "config_sha256": sha(model.cfg["_config_path"]),
                "baseline_sha256": sha(BASELINE), "source_manifest_sha256": sha(SOURCE_MANIFEST),
                "historical_protocols_unchanged": True})
