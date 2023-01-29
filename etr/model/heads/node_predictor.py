import torch.nn as nn

from etr.model.tree_model import Default_FNN
from etr.model.tools.allennlp import replace_masked_values, masked_log_softmax


class NodePredictor(nn.Module):

    def __init__(self, hidden_size, dropout_prob):
        super(NodePredictor, self).__init__()
        self.node_predictor = Default_FNN(hidden_size, hidden_size, 2, dropout_prob)

    def forward(self, input_vec, node_mask):

        # input_vec [batch, node_len, dim]/ [seq_len, dim]  mask [batch, seq_len]/ [seq_len]
        node_outputs = replace_masked_values(input_vec, node_mask.unsqueeze(-1), 0)
        node_predictions = self.node_predictor(node_outputs)
        node_predictions = masked_log_softmax(node_predictions, node_mask.unsqueeze(-1))
        node_predictions = replace_masked_values(node_predictions, node_mask.unsqueeze(-1), 0)
        return node_predictions