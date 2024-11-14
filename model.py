import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

def load_model(cfg):
    if cfg.model == "resnet50":
        model = torchvision.models.resnet50(weights=None)
        model.fc.out_features = cfg.num_classes

        if cfg.mode == 'simclr':
            dim_mlp = model.fc.in_features
            model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
    
    elif cfg.model == "efficientnet_b4":
        model = torchvision.models.efficientnet_b4(weights=None)
        model.classifier[1].out_features = cfg.num_classes
    
    elif cfg.model == "googlenet":
        model = model = torchvision.models.googlenet(weights=None)
        model.fc.out_features = cfg.num_classes
    
    elif cfg.model == "vitl16":
        model = torchvision.models.vit_l_16(weights=None)
        model.heads.head.out_features = cfg.num_classes

    else:
        raise NotImplementedError("This model is not implemented.")

    return model

class ClassifierEngine(pl.LightningModule):
    def __init__(self, cfg):
        super(ClassifierEngine, self).__init__()
        self.cfg = cfg
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
    
    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def training_step(self, batch, batch_idx):
        lr = self.optimizers().param_groups[0]["lr"]
        if self.cfg.mode == 'cls':
            img, target = batch
            output = self(img)
            if isinstance(output, torchvision.models.GoogLeNetOutputs):
                output = output.logits

            loss = self.criterion(output, target)
        elif self.cfg.mode == 'simclr':
            images, _ = batch           # ssl learning, we don't need labels
            images = torch.cat(images, dim=0)

            features = self(images)
            logit, labels = self.info_nce_loss(features)
            loss = self.criterion(logit, labels)

        else:
            raise NotImplementedError("This mode is not implemented.")

        
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log(
            "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        img, target = batch
        output = self(img)
        if isinstance(output, torchvision.models.GoogLeNetOutputs):
            output = output.logits

        loss = self.criterion(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        preds = {"val_loss" : loss, "correct" : correct}
        self.validation_step_outputs.append(preds) 
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('acc', correct, on_step=False, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        all_correct = sum([output["correct"] for output in self.test_step_outputs])
        avg_acc = all_correct / len(self.validation_step_outputs)
        self.log('accuracy', avg_acc, on_step=False, on_epoch=True)
        self.log('avg_val_loss', avg_loss, on_step=False, on_epoch=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        img, target = batch
        output = self(img)
        if isinstance(output, torchvision.models.GoogLeNetOutputs):
            output = output.logits
            
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
