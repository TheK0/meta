class TFGraphBatchIterable:
    def __init__(self, samples, max_num_nodes: int):
        self.samples = list(samples)
        self.max_num_nodes = max_num_nodes

    def __iter__(self):
        return iter(self.samples)
