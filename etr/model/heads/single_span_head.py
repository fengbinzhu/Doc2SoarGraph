import torch.nn as nn

from etr.model.tools.allennlp import masked_log_softmax, replace_masked_values

class SingleSpanHead(nn.Module):

    def __init__(self, input_size):
        super(SingleSpanHead, self).__init__()
        self.start_pos_predict = nn.Linear(input_size, 1)
        self.end_pos_predict = nn.Linear(input_size, 1)
        self.NLL = nn.NLLLoss(reduction="mean")

    def forward(self, input_vec, mask):
        # [batch_size, seq_len]
        start_logits = self.start_pos_predict(input_vec).squeeze(-1)
        end_logits = self.end_pos_predict(input_vec).squeeze(-1)

        start_log_probs = masked_log_softmax(start_logits, mask)
        end_log_probs = masked_log_softmax(end_logits, mask)
        start_log_probs = replace_masked_values(start_log_probs, mask, -1e7)
        end_log_probs = replace_masked_values(end_log_probs, mask, -1e7)
        return start_log_probs, end_log_probs



