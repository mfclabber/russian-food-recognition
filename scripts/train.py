from typing import Tuple, Dict, List

import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from time import time



def train_step(device: torch.device,
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               dataloader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module = None,
               scheduler: torch.optim.lr_scheduler = None):

  model.to(device)
  model.train()

  train_loss = 0
  torch.cuda.empty_cache()

  time_epoch_start = time()

  for batch, (images, annotations) in enumerate(dataloader):

    images = [torch.Tensor(image).to(device) for image in images]
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

    if criterion == None:
      loss_dict = model(images, annotations)
      loss = sum(loss for loss in loss_dict.values())
    else:
      pass
    train_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
      scheduler.step()

    # print(f"LOSSES: {loss}")
    # print(model(images[0]))

  train_loss = train_loss / len(dataloader)
  time_epoch_end = time() - time_epoch_start

  return train_loss, time_epoch_end



@torch.no_grad()
def validation_step(device: torch.device,
                    model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    criterion: torch.nn.Module = None):
  model.to(device)
  model.eval()

  val_loss = 0
  min_val_loss = 1e6
  torch.cuda.empty_cache()

  time_epoch_start = time()
  with torch.inference_mode():
    for batch, (images, annotations) in enumerate(dataloader):

      images = [torch.Tensor(image).to(device) for image in images]
      annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

      if criterion == None:
        model.train()
        loss_dict = model(images, annotations)
        # print(loss_dict)
        loss = sum(loss for loss in loss_dict.values())
        model.eval()
      else:
        pass
      val_loss += loss.item()

      # print(f"TESTING MODEL {model.predict(images)}")
      # break
      # print(f"LOSSES: {loss}")

  val_loss /= len(dataloader)
  time_epoch_end = time() - time_epoch_start

  # val_loss = validate(epoch)
  if val_loss < min_val_loss:
      print('NEW BEST MODEL!')
      torch.save(model.state_dict(), 'best_model.pth')
      min_val_loss = val_loss
  torch.save(model.state_dict(), 'latest_model.pth')

  return val_loss, time_epoch_end



def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device):
  for epoch in tqdm(range(epochs)):
    train_loss, train_train_epoch = train_step(model=model,
                                        dataloader=train_dataloader,
                                        optimizer=optimizer,
                                        device=device)

    print(f"EPOCH: {epoch+1} | TRAIN LOSS: {train_loss} | TRAIN TIME: {train_train_epoch}")

    val_loss, val_time_epoch = validation_step(model=model,
                                          dataloader=test_dataloader,
                                          device=device)

    print(f"EPOCH: {epoch+1} | VAL LOSS: {val_loss} | VAL TIME: {val_time_epoch}")