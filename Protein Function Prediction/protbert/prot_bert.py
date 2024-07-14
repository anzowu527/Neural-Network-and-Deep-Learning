import torch
import torchmetrics
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Adam, SGD, AdamW
from pytorch_lightning.callbacks import EarlyStopping

from transformers import BertTokenizer, BertGenerationEncoder

class ProteinClassifier(LightningModule):
    def __init__(self, n_classes=25):
        super().__init__()
        
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.embedder = BertGenerationEncoder.from_pretrained("Rostlab/prot_bert")
        #dmodel = 1024
        self.model = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(),             
            nn.Dropout(0.5),      
            nn.Linear(512, n_classes)  
        )       
        #self.model = nn.Linear(dmodel, n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        lengths = torch.tensor([len(i) for i in x]).to(self.device)
        ids = self.tokenizer(x, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(self.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self.device).to(self.dtype)
        with torch.no_grad():
            embeddings = self.embedder(input_ids=input_ids,
                                   attention_mask=attention_mask).last_hidden_state
        #embeddings = embeddings.sum(dim=1)/lengths.view(-1, 1)
        embeddings = embeddings.mean(dim=1)  # Pooling (mean)

        logits = self.model(embeddings)
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
        lr = 0.001 # I tried multiple lr and optimizers
        optimizer = SGD(self.parameters(), lr=lr, momentum=0.9)
        #optimizer = Adam(self.parameters(), lr=lr)
        #optimizer = AdamW(self.parameters(), lr=lr)
    
        return optimizer
    