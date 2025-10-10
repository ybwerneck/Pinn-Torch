from .dependencies import *
from .Loss import *

loss_map = {
    "MAE": lambda tgt, pred: torch.mean(torch.abs(tgt - pred)),
    "MSE": lambda tgt, pred: torch.mean((tgt - pred).ravel() ** 2),
    "RMSE": lambda tgt, pred: torch.mean(torch.sum(((tgt - pred)) ** 2, dim=1) ** 0.5),
    "KLDivergenceLoss": lambda tgt, pred: F.kl_div(pred, tgt),
    "CosineSimilarityLoss": lambda tgt, pred: 1 - F.cosine_similarity(pred, tgt),
    "LPthLoss": lambda tgt, pred, p: torch.mean(
        torch.pow(torch.sum(torch.pow(torch.abs(tgt - pred), p), axis=1), 1 / p)
    ),
}


class LOSS_PINN(torch.nn.Module):

    def __init__(
        self,
        device=torch.device("cuda"),
        criterium="RMSE",
        name="LossPinn",
        batch_size=10000,
    ):
        self.device = device
        self.criterium = loss_map[criterium]
        self.name = name
        self.batch_size = batch_size
        self.i = 0
        self.eval = False
        self.batchGen = False

        super(LOSS_PINN, self).__init__()

    def getBatch(self, data_in, target):
        batch = data_in[self.i : self.i + self.batch_size]
        tgt = target[self.i : self.i + self.batch_size]
        self.i += self.batch_size

        if self.i + self.batch_size > len(data_in):
            self.i = 0

        return batch.to(self.device), tgt.to(self.device)

    def setBatchGenerator(self, batch_generator, *batch_args):
        self.batch_generator = batch_generator
        self.batch_args = batch_args
        self.batchGen = False

    def setEvalFunction(self, eval_func, *eval_args):
        self.eval_func = eval_func
        self.eval_args = eval_args
        self.eval = True

    def forward(self, model, *loss_args):
        
        if self.batchGen:
            batch, tgt = self.batch_generator(
                self.batch_size, self.device, *self.batch_args
            )

        else:
            batch, tgt = self.getBatch()

        if self.eval:
            pred = self.eval_func(batch, model, *self.eval_args)

        else:
            pred = model(batch)

        return self.criterium(tgt, pred, *loss_args)


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
