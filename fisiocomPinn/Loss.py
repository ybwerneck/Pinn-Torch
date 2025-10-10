from .dependencies import *

loss_map = {
    "MAE": lambda tgt, pred: torch.mean(torch.abs(tgt - pred)),
    "MSE": lambda tgt, pred: torch.mean((tgt - pred).ravel() ** 2),
    "RMSE": lambda tgt, pred: torch.mean(torch.sum(((tgt - pred)) ** 2, dim=1) ** 0.5),
    "KLDivergenceLoss": lambda tgt, pred: F.kl_div(pred, tgt),
    "CosineSimilarityLoss": lambda tgt, pred: 1 - F.cosine_similarity(pred, tgt),
    "LPthLoss": lambda tgt, pred, p: torch.mean(
        torch.pow(torch.sum(torch.pow(torch.abs(tgt - pred), p), axis=1), 1 / p)
    ),
    "L2": lambda tgt, pred: torch.mean(
        torch.sqrt(torch.sum((pred - tgt) ** 2, dim=list(range(1, (pred - tgt).ndim))))
        / torch.sqrt(torch.sum(tgt**2, dim=list(range(1, tgt.ndim))) + 1e-8)
    ),
    "L2_squared": lambda tgt, pred: torch.mean(
        (torch.sum((pred - tgt) ** 2, dim=list(range(1, (pred - tgt).ndim))))
        / (torch.sum(tgt**2, dim=list(range(1, tgt.ndim))) + 1e-8)
    ),
}


class LOSS(torch.nn.Module):

    def __init__(
        self,
        device=torch.device("cuda"),
        criterium="RMSE",
        name="Loss",
        batch_size=10000,
    ):
        self.device = device
        self.criterium = loss_map[criterium]
        self.name = name
        self.batch_size = batch_size
        self.i = 0
        self.eval = False
        self.batchGen = False
        self.data_loss = False

        super(LOSS, self).__init__()

    def add_data(self, data_in, target):
        self.data_in = data_in
        self.target = target
        self.data_loss = True

    def getBatch(self):

        if self.data_loss:
            batch = self.data_in[self.i : self.i + self.batch_size]
            tgt = self.target[self.i : self.i + self.batch_size]
            self.i += self.batch_size

            if self.i + self.batch_size > len(self.data_in):
                self.i = 0

            return batch.to(self.device), tgt.to(self.device)

        else:
            print("Data not set")

            return None

    def setBatchGenerator(self, batch_generator, *batch_args):
        self.batch_generator = batch_generator
        self.batch_args = batch_args
        self.batchGen = True

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

        elif self.data_loss:
            pred = model(batch)

        else:
            print("None evaluation function presented")

        return self.criterium(tgt, pred, *loss_args)
