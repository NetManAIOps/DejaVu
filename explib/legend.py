from typing import Dict


class _LegendDB:
    def __init__(self):
        self._name_db: Dict[str, int] = {}
        # 数量最好是互质的
        self.colors = ['#d7191c', '#fdae61', '#ffff33', '#abdda4', '#2b83ba']
        self.hatches = ["", "///", "\\\\", "---"]
        self.linestyles = ['-', '-.', '--']
        self.markers = ['.', '+']

    def __getitem__(self, name: str) -> int:
        if name not in self._name_db:
            self._name_db[name] = len(self._name_db)
        return self._name_db[name]

    def clear(self):
        self._name_db = {}

    def get_legend_style(self, name: str) -> Dict[str, str]:
        return {
            "color": self.colors[self[name] % len(self.colors)],
            "linestyle": self.linestyles[self[name] % len(self.linestyles)],
            "marker": self.markers[self[name] % len(self.markers)],
        }

    def get_hatch_style(self, name: str) -> Dict[str, str]:
        return {
            "hatch": self.hatches[self[name] % len(self.hatches)],
            "color": self.colors[self[name] % len(self.colors)],
        }


legend_db = _LegendDB()

get_line_style = legend_db.get_legend_style
get_hatch_style = legend_db.get_hatch_style
