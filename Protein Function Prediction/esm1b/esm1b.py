import torch
import torchmetrics
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Adam
from torch.optim import SGD
from transformers import BertTokenizer, BertModel 

class ProteinClassifier(LightningModule):
    def __init__(self, n_classes=25):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("facebook/esm-1b", do_lower_case=False)
        self.embedder = BertModel.from_pretrained("facebook/esm-1b")
        #dmodel = 1024
        self.model = nn.Sequential(
            nn.Linear(1280, 512), 
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),            
            nn.Dropout(0.5),      
            nn.Linear(512, n_classes)  
        )       
        #self.model = nn.Linear(dmodel, n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=1280)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.embedder(**inputs)
        embeddings = outputs.last_hidden_state
        pooled_embeddings = embeddings.mean(dim=1)
        logits = self.model(pooled_embeddings)
        return logits
    
    def training_step(self, batch, batch_idx):
        '''
        calculate output --> loss --> training accuracy and save to self.log
        return loss
        '''
        x, y = batch 
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        '''
        Make predictions and calculate validation accuracy/F1 score and save to self.log
        '''
        x, y = batch
        logits = self(x)  
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)
        self.val_f1.update(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {
            "val_loss": loss,
            "val_acc": self.val_accuracy.compute(), 
            "val_f1": self.val_f1.compute(), 
        }

    def configure_optimizers(self):
        '''
        return optimizer for the model
        '''
        optimizer = SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer
    