
import logging
from plugin_system import PluginInterface, PluginInfo
logger = logging.getLogger("tp")
class CounterPlugin(PluginInterface):
    def __init__(self, config=None):
        super().__init__(config); self.count = 0
    def get_info(self):
        return PluginInfo(name="counter", version="1.0", description="计数", author="t", entry_point="CounterPlugin")
    def execute(self, params, context=""):
        self.count += 1; return {"count": self.count}
    def initialize(self): return True
    def shutdown(self): pass
