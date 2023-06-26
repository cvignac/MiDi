import torch
import torchmetrics
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric, MeanSquaredError

from midi.utils import PlaceHolder

# TODO: these full check updates are probably not necessary

class SumExceptBatchMetric(Metric):
    full_state_update = True

    def __init__(self):
        super().__init__(sync_on_compute=False, dist_sync_on_step=False)
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, values) -> None:
        self.total_value += torch.sum(values)
        self.total_samples += values.shape[0]

    def compute(self):
        return self.total_value / max(self.total_samples, 1)


class SumExceptBatchMSE(MeanSquaredError):
    full_state_update = True

    def __init__(self):
        super().__init__(sync_on_compute=False, dist_sync_on_step=False)

    def update(self, preds: Tensor, target: Tensor) -> None:
        assert preds.shape == target.shape, f"Preds has shape {preds.shape} while target has shape {target.shape}"
        sum_squared_error, n_obs = self._mean_squared_error_update(preds, target)

        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def _mean_squared_error_update(self, preds: Tensor, target: Tensor):
            """ Updates and returns variables required to compute Mean Squared Error. Checks for same shape of input
            tensors.
                preds: Predicted tensor
                target: Ground truth tensor
            """
            diff = preds - target
            sum_squared_error = torch.sum(diff * diff)
            n_obs = preds.shape[0]
            return sum_squared_error, n_obs


class SumExceptBatchKL(torchmetrics.KLDivergence):
    def __init__(self):
        super().__init__(log_prob=False, reduction='sum', sync_on_compute=False, dist_sync_on_step=False)

    def update(self, p, q) -> None:
        p = p.flatten(start_dim=1)
        q = q.flatten(start_dim=1)
        super().update(p, q)


class CrossEntropyMetric(Metric):
    full_state_update = True

    def __init__(self):
        super().__init__(sync_on_compute=False, dist_sync_on_step=False)
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        target = torch.argmax(target, dim=-1)
        output = F.cross_entropy(preds, target, reduction='sum')
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples


class ProbabilityMetric(Metric):
    def __init__(self):
        """ This metric is used to track the marginal predicted probability of a class during training. """
        super().__init__(sync_on_compute=False, dist_sync_on_step=False)
        self.add_state('prob', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor) -> None:
        self.prob += preds.sum()
        self.total += preds.numel()

    def compute(self):
        return self.prob / self.total


class NLL(Metric):
    full_state_update = True
    def __init__(self):
        super().__init__(sync_on_compute=False, dist_sync_on_step=False)
        self.add_state('total_nll', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, batch_nll) -> None:
        self.total_nll += torch.sum(batch_nll)
        self.total_samples += batch_nll.numel()

    def compute(self):
        return self.total_nll / max(self.total_samples,1)


class PosMSE(SumExceptBatchMSE):
    def update(self, preds: PlaceHolder, target: PlaceHolder):
        super().update(preds.pos, target.pos)


class XKl(SumExceptBatchKL):
    def update(self, preds: PlaceHolder, target: PlaceHolder):
        super().update(preds.X, target.X)

class ChargesKl(SumExceptBatchKL):
    def update(self, preds: PlaceHolder, target: PlaceHolder):
        super().update(preds.charges, target.charges)


class EKl(SumExceptBatchKL):
    def update(self, preds: PlaceHolder, target: PlaceHolder):
        super().update(preds.E, target.E)


class YKl(SumExceptBatchKL):
    def update(self, preds: PlaceHolder, target: PlaceHolder):
        super().update(preds.y, target.y)


class PosLogp(SumExceptBatchMetric):
    def update(self, preds, target):
        # TODO
        return -1


class XLogP(SumExceptBatchMetric):
    def update(self, preds, target):
        super().update(target.X * preds.X.log())


class ChargesLogp(SumExceptBatchMetric):
    def update(self, preds, target):
        super().update(target.charges * preds.charges.log())


class ELogP(SumExceptBatchMetric):
    def update(self, preds, target):
        super().update(target.E * preds.E.log())


class YLogP(SumExceptBatchMetric):
    def update(self, preds, target):
        super().update(target.y * preds.y.log())
