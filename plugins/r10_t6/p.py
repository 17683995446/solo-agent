from plugin_system import PluginInterface, PluginInfo
class P(PluginInterface):
    def __init__(self, c=None): super().__init__(c)
    def get_info(self): return PluginInfo(name="test", version="2", description="x", author="t", entry_point="P")
    def execute(self, p, c=""): return {"v": 2}
    def initialize(self): return True