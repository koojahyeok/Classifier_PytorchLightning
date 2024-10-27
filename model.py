import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

def load_model(cfg):
    if cfg.model == "resnet50":
        model = torchvision.models.resnet50(weights=None)
        model.fc.out_features = cfg.num_classes
    
    elif cfg.model == "efficientnet_b4":
        model = torchvision.models.efficientnet_b4(weights=None)
        model.classifier[1].out_features = cfg.num_classes

    else:
        raise NotImplementedError("This model is not implemented.")

    return model

class ClassifierEngine(pl.LightningModule):
    def __init__(self, cfg):
        super(ClassifierEngine, self).__init__()
        self.model = load_model(cfg)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay
        self.milestones = cfg.milestones
        self.gamma = cfg.gamma
        self.last_epoch = cfg.last_epoch
        self.scheduler_verbosity = cfg.scheduler_verbosity
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        lr = self.optimizers().param_groups[0]["lr"]
        img, target = batch
        output = self(img)
        loss = self.criterion(output, target)
        self.log('train_loss', loss)
        self.log(
            "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        img, target = batch
        output = self(img)
        loss = self.criterion(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        preds = {"val_loss" : loss, "correct" : correct}
        self.validation_step_outputs.append(preds) 
        self.log('val_loss', loss)
        self.log('acc', correct)
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        all_correct = sum([output["correct"] for output in self.test_step_outputs])
        avg_acc = all_correct / len(self.validation_step_outputs)
        self.log('accuracy', avg_acc)
        self.log('avg_val_loss', avg_loss)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        img, target = batch
        output = self(img)
        loss = self.criterion(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()/ len(target)
        preds  = {"correct": correct}
        self.test_step_outputs.append(preds)
        self.log('test_loss', loss)
        return loss
    
    def on_test_epoch_end(self):
        all_correct = sum([output["correct"] for output in self.test_step_outputs])
        accuracy = all_correct / len(self.test_step_outputs)

        self.log("accuracy", accuracy)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                        self.milestones,
                                                        self.gamma,
                                                        last_epoch=self.last_epoch, 
                                                        verbose=self.scheduler_verbosity)
        return [optimizer], [scheduler]
