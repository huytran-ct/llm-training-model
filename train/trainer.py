from tqdm import tqdm
import torch
class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device: str, batch_size: int, num_epochs: int, interval_eval:int=1) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.interval_eval = interval_eval
        self.device = device

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.to(self.device)
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            print(f"Epoch {epoch}: Train loss: {total_loss}, learning rate: {self.optimizer.param_groups[0]['lr']}")
            if epoch % self.interval_eval == 0:
                self.eval()

    def eval(self):
        self.model.to(self.device)
        self.model.eval()
        eval_loss = 0
        for step, batch in enumerate(tqdm(self.eval_dataloader)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs =self.model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
        print("Eval loss: ", eval_loss)
