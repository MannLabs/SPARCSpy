import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import matplotlib.pyplot as plt
import PIL
import numpy as np
import matplotlib.ticker as ticker


from vipercore.ml.models import GolgiVGG, GolgiCAE

import gc
    
import sys
sys.path.append("/home/default/projects/opticalScreening/viper/machine_learning")

class MultilabelSupervisedModel(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        
        
        self.save_hyperparameters()
            
        self.network = GolgiVGG(in_channels=self.hparams["num_in_channels"],
                                cfg = "B",
                                dimensions=128,
                                num_classes=self.hparams["num_classes"])

        
        self.train_metrics = torchmetrics.MetricCollection([torchmetrics.Precision(average="none",num_classes=self.hparams["num_classes"]), 
                                                            torchmetrics.Recall(average="none",num_classes=self.hparams["num_classes"]),
                                                            torchmetrics.Accuracy(average=None,num_classes=self.hparams["num_classes"]),
                                                            torchmetrics.ConfusionMatrix(num_classes=self.hparams["num_classes"], normalize="true")]) 
        
        self.val_metrics = torchmetrics.MetricCollection([torchmetrics.Precision(average="none",num_classes=self.hparams["num_classes"]), 
                                                          torchmetrics.Recall(average="none",num_classes=self.hparams["num_classes"]),
                                                          torchmetrics.Accuracy(average=None,num_classes=self.hparams["num_classes"])])
        
        self.test_metrics = torchmetrics.MetricCollection([torchmetrics.Precision(average="none",num_classes=self.hparams["num_classes"]), 
                                                          torchmetrics.Recall(average="none",num_classes=self.hparams["num_classes"]),
                                                          torchmetrics.Accuracy(average=None,num_classes=self.hparams["num_classes"])])
        
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"precision/train": 0,"recall/train": 0,"precision/val": 0,"recall/val": 0})
    
    def forward(self, x):
        
        return self.network(x)
    
    def configure_optimizers(self):
        if self.hparams["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams["learning_rate"])
        elif self.hparams["optimizer"] == "Adam":
            #set weight decay to 0 if not specified in hparams 
            if self.hparams["weight_decay"] is None:
                self.hparams["weight_decay"] = 0
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"], weight_decay=self.hparams["weight_decay"])
        else:
            raise ValueError("No optimizier specified in hparams")
        return optimizer
    
    def on_train_epoch_start(self):
        pass
        
    def on_validation_epoch_start(self):
         pass
        
    def confusion_plot(self, matrix):
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)       
        cax = ax.matshow(matrix,cmap="magma")    

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        
        ax.set_xticklabels([''] +self.hparams["class_labels"], rotation = -45)
        ax.set_yticklabels([''] +self.hparams["class_labels"])
        
        ax.set_xlabel('prediction')
        ax.set_ylabel('ground truth')
    
        fig.tight_layout()
        
        fig.canvas.draw()
        
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = np.expand_dims(data, axis=0)
        
        return data
        
    def on_train_epoch_end(self):
        
        metrics = self.train_metrics.compute()
        

        img = self.confusion_plot(metrics["ConfusionMatrix"].detach().cpu())
        self.logger.experiment.add_image('confusion', img,self.current_epoch, dataformats="NHWC") 
        
        for i, label in enumerate(self.hparams["class_labels"]):
        
            self.log("precision_train/{}".format(label), metrics["Precision"][i])
            self.log("recall_train/{}".format(label), metrics["Recall"][i])
            self.log("accurac_train/{}".format(label), metrics["Accuracy"][i])
        
        # Reseting internal state such that metric ready for new data
        self.train_metrics.reset()

    
    def on_validation_epoch_end(self):
        
        metrics = self.val_metrics.compute()
        
        for i, label in enumerate(self.hparams["class_labels"]):
        
            self.log("precision_val/{}".format(label), metrics["Precision"][i])
            self.log("recall_val/{}".format(label), metrics["Recall"][i])
            self.log("accurac_val/{}".format(label), metrics["Accuracy"][i])
        
        # Reseting internal state such that metric ready for new data
        self.val_metrics.reset()
        
    
    def training_step(self, batch, batch_idx):
        data, label = batch
        
        data, label = data.cuda(), label.cuda()
        
        output = self.network(data)
        loss = F.nll_loss(output, label)
        
        # log accuracy metrics
        non_log = torch.exp(output)
        
        self.train_metrics(non_log, label)
        
        
        self.log('loss/train', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, label = batch
        
        data, label = data.cuda(), label.cuda()
        
        output = self.network(data)
        loss = F.nll_loss(output, label)

        #accuracy metrics
        non_log = torch.exp(output)    
        
        self.val_metrics(non_log, label)
        
        self.log('loss/val', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        data, label = batch
        data, label = data.cuda(), label.cuda()

        output = self.network(data)
        loss = F.nll_loss(output, label)

        non_log = torch.exp(output)    
        self.test_metrics(non_log, label)
        self.log('loss/test', loss, prog_bar=True)

    def test_end(self):
        #use same metrics as in validation
        metrics = self.test_metrics.compute()
        
        for i, label in enumerate(self.hparams["class_labels"]):
            self.log("precision_test/{}".format(label), metrics["Precision"][i])
            self.log("recall_test/{}".format(label), metrics["Recall"][i])
            self.log("accurac_test/{}".format(label), metrics["Accuracy"][i])
        
        # Reseting internal state such that metric ready for new data
        self.test_metrics.reset()
        
class GeneralModel(pl.LightningModule):

    def __init__(self, model, hparams):
        super().__init__()
            
        print(hparams)
        self.hp = hparams
        self.network = model
        
        self.train_metrics = torchmetrics.MetricCollection([torchmetrics.Precision(average="none",num_classes=hparams["num_classes"]), 
                                                            torchmetrics.Recall(average="none",num_classes=hparams["num_classes"]),
                                                            torchmetrics.AUROC(num_classes=hparams["num_classes"]),
                                                           torchmetrics.Accuracy()]) 
        
        self.val_metrics = torchmetrics.MetricCollection([torchmetrics.Precision(average="none",num_classes=hparams["num_classes"]), 
                                                          torchmetrics.Recall(average="none",num_classes=hparams["num_classes"]),
                                                          torchmetrics.AUROC(num_classes=hparams["num_classes"]),
                                                          torchmetrics.Accuracy()])
        
        #
        
    def on_train_start(self):
        self.logger.log_hyperparams(self.hp, {"precision/train": 0,"recall/train": 0,"precision/val": 0,"recall/val": 0})
    
    def forward(self, x):
        
        return self.network(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hp["lr"])
        return optimizer
    
    def on_train_epoch_start(self):
        pass
        
    def on_validation_epoch_start(self):
         pass
        
    def on_train_epoch_end(self,outputs):
        metrics = self.train_metrics.compute()
        
        self.log("precision/train", metrics["Precision"][0])
        self.log("recall/train", metrics["Recall"][0])
        self.log("auroc/train", metrics["AUROC"])
        self.log("accuracy/train", metrics["Accuracy"])
        
        # Reseting internal state such that metric ready for new data
        self.train_metrics.reset()
    
    def on_validation_epoch_end(self):
        
        metrics = self.val_metrics.compute()
        
        self.log("precision/val", metrics["Precision"][0])
        self.log("recall/val", metrics["Recall"][0])
        self.log("auroc/val", metrics["AUROC"])
        self.log("accuracy/val", metrics["Accuracy"])
        
        # Reseting internal state such that metric ready for new data
        self.val_metrics.reset()
    
    def training_step(self, batch, batch_idx):
        data, label = batch
        
        data, label = data.cuda(), label.cuda()
        
        output = self.network(data)
        loss = F.nll_loss(output, label)
        
        # log accuracy metrics
        non_log = torch.exp(output)
        
        self.train_metrics(non_log, label)
        
        
        self.log('loss/train', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, label = batch
        data, label = data.cuda(), label.cuda()
        
        output = self.network(data)
        loss = F.nll_loss(output, label)
    

        #accuracy metrics
        non_log = torch.exp(output)    
        
        self.val_metrics(non_log, label)
        
        self.log('loss/val', loss, prog_bar=True)

class AutoEncoderModel(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
            
        print(self.hparams)
        self.save_hyperparameters()

        self.network = GolgiCAE(encoder_cfg = self.hparams["encoder_cfg"],
                decoder_cfg = self.hparams["decoder_cfg"],
                in_channels = self.hparams["num_in_channels"],
                out_channels = self.hparams["num_out_channels"])
        
        self.loss = torch.nn.BCELoss()
        
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        
    def forward(self, x):
        return self.network(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
        
        return optimizer
    
    def on_train_epoch_start(self):
        pass
        
    def on_validation_epoch_start(self):
         pass
        
    def on_train_epoch_end(self,outputs):
        
        pass
        #tensorboard = self.logger.experiment
        #tensorboard.add_image()
        
    def training_step(self, batch, batch_idx):     
        
        opt = self.optimizers()
        opt.zero_grad()
    
        images, _ = batch

        images = images.cuda()
        input_golgi_channel = images[:,3:4,:,:]
        
        output = self.network(images)
        
        
        loss = self.loss(output, input_golgi_channel)    
        
        self.manual_backward(loss, retain_graph=False)
        
        opt.step()
        loss = loss.detach()
        
        self.log('loss/train', loss, prog_bar=True)
        
        if batch_idx > 118:
            torch.cuda.empty_cache()
            
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass
        

        
    def validation_step(self, batch, batch_idx):     
        
        opt = self.optimizers()
        opt.zero_grad()
    
        images, labels = batch

        images = images.cuda()
        input_golgi_channel = images[:,3:4,:,:]
        
        output = self.network(images)
        
        
        loss = self.loss(output, input_golgi_channel)    

        loss = loss.detach()
        
        if batch_idx == 0:
            SAMPLE_NUM=10
            
            input_sample = images[:SAMPLE_NUM,:,:,:].detach().cpu()
            output_sample = output[:SAMPLE_NUM,:,:,:].detach().cpu()
            label_sample = labels[:SAMPLE_NUM].detach().cpu()
            
            img = self.sample_plot(input_sample,output_sample,label_sample)
            self.logger.experiment.add_image('prediction', img, self.current_epoch, dataformats="NHWC") 
        
        print(batch_idx, self.current_epoch)
        
        self.log('loss/val', loss.item(), prog_bar=True)
        return loss
    
    def sample_plot(self, input_sample, output_sample, label_sample):
        
        fig, axs = plt.subplots(6, len(label_sample), figsize=(15, 10))
        
        for i in range(len(label_sample)):
    
            label = int(label_sample[i].item())

           
            channels = input_sample[i]

            axs[0,i].set_title(self.hparams["class_labels"][label])

            for j in range(6):

                if j <2:
                    color = "gray"
                else:
                    color = "magma"

                if j < 5:
                    c = channels[j]

                else:
                    c = output_sample[i,0]

                axs[j,i].imshow(c, cmap=color)
                axs[j,i].xaxis.set_ticks([])
                axs[j,i].yaxis.set_ticks([])

        rows = ["Cell Mask", "Nucleus Mask", "Nucleus", "TGOLN2-mCherry","WGA","CAE"]
        
        for ax, row in zip(axs[:,0], rows):
            ax.set_ylabel(row, rotation=0, size='large', ha="right")

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
        fig.tight_layout()
        
        fig.canvas.draw()
        
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = np.expand_dims(data, axis=0)
        
        return data

class AEModel(pl.LightningModule):

    def __init__(self, model, hparams):
        super().__init__()
            
        print(hparams)
        self.hp = hparams
        self.network = model
        self.loss = torch.nn.BCELoss()
        
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        
    def forward(self, x):
        return self.network(x)
    
    def configure_optimizers(self):
        optimizer = optimizer = torch.optim.Adam(self.parameters(), lr=self.hp["lr"])
        
        return optimizer
    
    def on_train_epoch_start(self):
        pass
        
    def on_validation_epoch_start(self):
         pass
        
    def on_train_epoch_end(self,outputs):
        
        pass
        #tensorboard = self.logger.experiment
        #tensorboard.add_image()
        
    def training_step(self, batch, batch_idx):     
        
        opt = self.optimizers()
        opt.zero_grad()
    
        images, _ = batch

        images = images.cuda()
        input_golgi_channel = images[:,3:4,:,:]
        
        output = self.network(images)
        
        
        loss = self.loss(output, input_golgi_channel)    
        self.manual_backward(loss)
        
        opt.step()
        loss = loss.detach()
        
        self.log('loss/train', loss, prog_bar=True)
        
