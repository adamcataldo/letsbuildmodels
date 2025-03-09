from matplotlib import pyplot as plt
import numpy as np
import lbm.workflow.core as core 
import lbm.metrics as metrics
import matplotlib.ticker as mtick


def train_and_validate(model, train_loader, val_loader, optimizer, loss_fn, 
                       epochs, device='cpu', metrics=[]):
    t_loss, v_loss = core.train_and_validate(model, train_loader, val_loader, 
                                             optimizer, loss_fn, epochs, 
                                             device=device, metrics=metrics)
    plt.plot(np.arange(1, len(t_loss) + 1), t_loss, label='Training',
             color='blue')
    plt.plot(np.arange(1, len(v_loss) + 1), v_loss, label='Validation', 
             color='orange')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    for metric in metrics:
        if hasattr(metric, 'show') and callable(metric.show):
            metric.show()
        else:
            print(f'{metric.__class__.__name__}: {metric.compute()}')
                                                    
class ReturnDeltas(metrics.ReturnDeltas):
    def show(self):
        errors = self.compute()
        # Plot of return errors, using unconnected dots to mark each error
        plt.hist(errors, bins=30)
        plt.xlabel('Return Error')
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.gca().tick_params(axis='y', labelleft=False)
        plt.show()


class DirectionalAccuracy(metrics.DirectionalAccuracy):
    def show(self):
        print(f'Directional Accuracy: {self.compute():.1%}')