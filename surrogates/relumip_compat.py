"""
relumip.AnnModel with a fix for its node-redundancy check.

relumip 1.x loops the check over every layer including the linear output
layer, but redundancy_matrix only has entries for the hidden layers. Any
network whose output bound has one sign (e.g. a flow or a power, always
> 0) then raises IndexError in embed_network_formulation. The output layer
has no ReLU, so it is simply skipped here.
"""

from relumip import AnnModel as _AnnModel


class AnnModel(_AnnModel):
    def _node_redundancy_check(self) -> None:
        p = self.networkParam
        for i in range(1, p.n_layers - 1):          # hidden layers only
            for j in range(p.nodes_per_layer[i]):
                if p.M_plus[i - 1][j] <= 0:
                    p.redundancy_matrix[i - 1][j] = -1
                elif p.M_minus[i - 1][j] >= 0:
                    p.redundancy_matrix[i - 1][j] = 1
