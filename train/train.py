import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Callable
from tqdm import tqdm


def logging(loss,
            score=None,
            mode='train',
            wandb_run=None,
            to_print=True):
        
    if to_print:
        print('Loss: {:.6f}\tScore: {}'.format(
            loss.item(),
            score))

    if wandb_run is not None:
        wandb_run.log({
            f'{mode}_loss': loss.item(),
            f'{mode}_score': score
        })


def evaluate(model,
             criterion,
             loader,
             device='cpu',
             eval_metric=None,
            ):
    model.eval()
    losses = []
    scores = []
    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            # y_batch = y_batch.unsqueeze(1).to(torch.float32)

            losses.append(criterion(output, y_batch))
            if eval_metric is not None:
                scores.append(eval_metric(y_batch.cpu().detach().numpy(), 
                                            output.argmax(1).cpu().detach().numpy()))
    return np.mean(losses), np.mean(scores)


def log_during_train(
    model: torch.nn.Module,
    output: torch.Tensor,
    y_batch: torch.Tensor,
    criterion: torch.nn.Module,
    loss: torch.Tensor,
    epoch: int, 
    batch_idx: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    device: str = 'cpu',
    max_epochs: int = 1,
    eval_metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
    wandb_run=None
    ) -> None:
    """
    Logs the loss and score during the training process.

    Args:
        model: The model to evaluate.
        output: The output of the model.
        y_batch: The labels of the batch.
        criterion: The loss function.
        loss: The loss of the model.
        epoch: The current epoch.
        batch_idx: The current batch index.
        train_loader: The data loader for the training data.
        val_loader: The data loader for the validation data.
        device: The device to use for evaluation.
        max_epochs: The maximum number of epochs to train for.
        eval_metric: The metric to use for evaluation.
        wandb_run: The W&B run to log to.
    """
    model.eval()
    score = np.nan
    if eval_metric is not None:
        score = eval_metric(y_batch.cpu().detach().numpy(), 
                            output.argmax(1).cpu().detach().numpy())
    with torch.no_grad():
        print('Train Epoch {}/{}  Batch {}/{}:\t'.format(epoch, max_epochs, batch_idx, len(train_loader)))
        logging(loss,
                score=score,
                mode='train',
                wandb_run=wandb_run,
                to_print=True)


    if val_loader is not None:
        model.eval()
        with torch.no_grad():
            print('Validation:\t')
            loss, score = evaluate(model,
                    criterion,
                    val_loader,
                    device=device,
                    eval_metric=eval_metric,
                    )
            logging(loss,
                    score=score,
                    mode='val',
                    wandb_run=wandb_run,
                    to_print=True)

            

def train(model: nn.Module,
          criterion: Callable,
          optimizer: torch.optim.Optimizer,
          train_loader: torch.utils.data.DataLoader,
          val_loader: Optional[torch.utils.data.DataLoader] = None,
          device: str = 'cpu',
          max_epochs: int = 1,
          eval_every: int = 1,
          eval_metric: Optional[Callable] = None,
          wandb_run = None
        ) -> None:
    """
    Train the model.

    Args:
        model (nn.Module): The model to train.
        criterion (Callable): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        train_loader (torch.utils.data.DataLoader): The data loader for the training data.
        val_loader (Optional[torch.utils.data.DataLoader]): The data loader for the validation data.
        device (str): The device to use for training.
        max_epochs (int): The maximum number of epochs to train for.
        eval_every (int): The frequency of evaluation during training.
        eval_metric (Optional[Callable]): The metric to use for evaluation.
        wandb_run (Optional[wandb.Run]): The W&B run to log to.

    Returns:
        None
    """
    train_history = {'loss': [], 'score': []}
    val_history = {'loss': [], 'score': []}

    model.train()
    for epoch in range(max_epochs):
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)

            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            if batch_idx % eval_every == 0:
                log_during_train(
                     model,
                     output,
                     y_batch,
                     criterion,
                     loss,
                     epoch, 
                     batch_idx,
                     train_loader,
                     val_loader,
                     device,
                     max_epochs,
                     eval_metric,
                     wandb_run
                )

                model.train()
                print()
