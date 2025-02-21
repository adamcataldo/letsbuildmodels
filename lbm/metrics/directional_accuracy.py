from torcheval.metrics.metric import Metric
import torch

class DirectionalAccuracy(Metric):
    def __init__(self, device = None) -> None:
        super().__init__(device=device)
        self._add_state("final_inputs", torch.tensor([], device=self.device)) 
        self._add_state("actual_outputs", torch.tensor([], device=self.device))
        self._add_state("predicted_outputs", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, input_sequences, actual_outputs, predicted_outputs):
        # Assumes the value of interest is always the first in/out dimension.
        # input_sequences: (seq_len, batch_size, input_dim)
        # t_actuals: (batch_size, output_dim)
        # t_predictions: (batch_size, output_dim)
        self.final_inputs = torch.cat((self.final_inputs, 
                                       input_sequences[-1, :, 0]))   
        self.actual_outputs = torch.cat((self.actual_outputs, 
                                         actual_outputs[:, 0]))
        self.predicted_outputs = torch.cat((self.predicted_outputs, 
                                            predicted_outputs[:, 0]))
        return self

    @torch.inference_mode()
    def compute(self):
        correct = ((self.actual_outputs > self.final_inputs) & 
                   (self.predicted_outputs > self.final_inputs)) | \
                  ((self.actual_outputs <= self.final_inputs) & 
                   (self.predicted_outputs <= self.final_inputs))
        return torch.mean(correct.float()).item()

    @torch.inference_mode()
    def merge_state(self, metrics):
        final_inputs = [self.final_inputs, ]
        actual_outputs = [self.actual_outputs, ]
        predicted_outputs = [self.predicted_outputs, ]
        for metric in metrics:
            final_inputs.append(metric.final_inputs)
            actual_outputs.append(metric.actual_outputs)
            predicted_outputs.append(metric.predicted_outputs)
        self.final_inputs = torch.cat(final_inputs)
        self.actual_outputs = torch.cat(actual_outputs)
        self.predicted_outputs = torch.cat(predicted_outputs)
        return self