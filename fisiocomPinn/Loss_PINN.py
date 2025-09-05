from .dependencies import *
from .Loss import *

loss_map = {
    "MAE": MAE,
    "MSE": MSE,
    "RMSE": RMSE,
    "KLDivergenceLoss": KLDivergenceLoss,
    "CosineSimilarityLoss": CosineSimilarityLoss,
    "LPthLoss": LPthLoss,
}


class LOSS_PINN(torch.nn.Module):

    def __init__(
        self,
        batch_size=10000,
        device=torch.device("cuda"),
        loss="RMSE",
        name="LossPinn",
    ):
        self.batch_size = batch_size
        self.device = device
        self.loss = loss
        self.name = name
        super(LOSS_PINN, self).__init__()

    def setBatchGenerator(self, batch_generator, *batch_args):
        self.batch_generator = batch_generator
        self.batch_args = batch_args

    def setPinnFunction(self, pinn_func, *pinn_args):
        self.pinn_func = pinn_func
        self.pinn_args = pinn_args

    def forward(
        self,
        model,
    ):
        batch, tgt = self.batch_generator(
            self.batch_size, self.device, *self.batch_args
        )

        criterium = loss_map[self.loss](batch, tgt)

        pred = model(batch)

        pinn_pred = self.pinn_func(pred, batch, *self.pinn_args)

        return criterium.loss(tgt, pinn_pred)


class LOSS_INITIAL(torch.nn.Module):

    def __init__(
        self,
        batch_size=10000,
        device=torch.device("cuda"),
        loss="RMSE",
        name="LossInital",
    ):
        self.batch_size = batch_size
        self.device = device
        self.loss = loss
        self.name = name
        super(LOSS_INITIAL, self).__init__()

    def setBatchGenerator(self, batch_generator, *args):
        self.batch_generator = batch_generator
        self.batch_args = args

    def forward(
        self,
        model,
    ):

        batch, tgt = self.batch_generator(
            self.batch_size, self.device, *self.batch_args
        )

        criterium = loss_map[self.loss](batch, tgt)

        pred = model(batch)

        return criterium.loss(tgt, pred)
