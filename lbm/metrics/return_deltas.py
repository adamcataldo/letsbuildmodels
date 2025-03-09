from torcheval.metrics.metric import Metric
import torch

class ReturnDeltas(Metric):
    def __init__(self, device = None) -> None:
        super().__init__(device=device)
        self._add_state("final_inputs", torch.tensor([], device=self.device)) 
        self._add_state("actual_outputs", torch.tensor([], device=self.device))
        self._add_state("predicted_outputs", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, input_sequences, actual_outputs, predicted_outputs):
        # Assumes the value of interest is always the first in/out dimension.
        # input_sequences: (batch_size, seq_length, input_dim)
        # actual_outputs: (batch_size, output_dim)
        # predicted_outputs: (batch_size, output_dim)
        self.final_inputs = torch.cat((self.final_inputs, 
                                       input_sequences[:, -1, 0]))   
        self.actual_outputs = torch.cat((self.actual_outputs, 
                                         actual_outputs[:, 0]))
        self.predicted_outputs = torch.cat((self.predicted_outputs, 
                                            predicted_outputs[:, 0]))
        return self

    @torch.inference_mode()
    def compute(self):
        predicted_returns = self.predicted_outputs / self.final_inputs - 1.0
        actual_returns = self.actual_outputs / self.final_inputs - 1.0
        return (predicted_returns - actual_returns).cpu().numpy()
    
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