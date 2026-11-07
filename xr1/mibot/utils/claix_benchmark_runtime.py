"""Two-way-only supervision; the historical synchronous runtime stays intact."""
import os

from mibot.utils.claix_process import process_alive, writer_lifetime_lock
from mibot.utils.claix_runtime import ProductionRuntime


class BenchmarkSyncRuntime(ProductionRuntime):
    def __init__(self, directory, *, run_root, **kwargs):
        super().__init__(directory, **kwargs)
        self._lifetime_lock = writer_lifetime_lock(run_root)

    def _close(self):
        if self._lifetime_lock is not None:
            self._lifetime_lock.close()
            self._lifetime_lock = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        controller = os.environ.get('XR1_CONTROLLER_ID')
        if controller and not process_alive(controller):
            self.stop.reason = 'controller_lost'
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_fit_end(self, trainer, pl_module):
        try:
            super().on_fit_end(trainer, pl_module)
        finally:
            self._close()

    def on_exception(self, trainer, pl_module, exception):
        try:
            super().on_exception(trainer, pl_module, exception)
        finally:
            self._close()
