import torch.nn as nn

from etr.model.tree_model import Default_FNN
from etr.model.tools.allennlp import replace_masked_values, masked_log_softmax


class SequenceTagHead(nn.Module):

    def __init__(self, hidden_size, dropout_prob):
        super(SequenceTagHead, self).__init__()
        self.tag_predictor = Default_FNN(hidden_size, hidden_size, 3, dropout_prob) # BIO
        self.NLLLoss = nn.NLLLoss(reduction="mean")

    def forward(self, input_vec, masks):

        # input_vec [batch, seq_len, dim]/ [seq_len, dim]  mask [batch, seq_len]/ [seq_len]
        sequence_output = replace_masked_values(input_vec, masks.unsqueeze(-1), 0)
        tag_predictions = self.tag_predictor(sequence_output)
        tag_pred_log_probs = masked_log_softmax(tag_predictions, masks.unsqueeze(-1))
        tag_pred_log_probs = replace_masked_values(tag_pred_log_probs, masks.unsqueeze(-1), 0)

        return tag_pred_log_probs