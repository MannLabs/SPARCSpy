import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

class GolgiCAE(nn.Module):
    
    
    en_cfgs = {
        'A': [16, "M", 32, "M", 64, "M", 128, "M", 256, "M", 512, "M"],
        'A_deep': [32, "M", 64, "M", 128, "M", 256, "M", 512, 512, "M", 512, 512, "M"],
        'B': [16, "M", 32, "M", 64, "M", 128, "M", 256, "M", 256, "M"],
        'C': [16, "M", 32, "M", 64, "M", 128, "M", 256, "M", 512, "M",512, "M"],
        'D': [16, "M", 32, "M", 64, "M", 128, "M", 256, "M", 256, "M",256, "M"],
        'E': [16, "M", 32, "M", 64, "M", 128, "M", 256, "M", 256, "M",128, "M"],
        }
    
    de_cfgs = {
        "A": [512, "U", 256, "U", 128, "U", 64, "U", 32, "U", 16, "U", 16],
        "A_deep": [512,512, "U", 512,512, "U", 256, "U", 128, "U", 64, "U", 32, "U", 16],
        "B": [256, "U", 256, "U", 128, "U", 64, "U", 32, "U", 16, "U", 16],      
        "C": [512, "U", 512, "U", 256, "U", 128, "U", 64, "U", 32, "U", 16, "U", 16],
        "D": [256, "U", 256, "U", 256, "U", 128, "U", 64, "U", 32, "U", 16, "U", 16],
        "E": [128, "U", 256, "U", 256, "U", 128, "U", 64, "U", 32, "U", 16, "U", 16],
    }
    
    latent_dim = {
        "A": 2048,
        "A_deep": 2048,
        "B": 1024,
        "C": 512,
        "D": 256,
        "E": 128,
    }
    
    initial_decoder_depth = {
        "A": 512,
        "A_deep": 512,
        "B": 256,
        "C": 512,
        "D": 256,
        "E": 128,
    }
    
    def __init__(self,
                cfg = "B",
                in_channels = 5,
                out_channels = 5,
                
                ):
        
        super(GolgiCAE, self).__init__()
        
        self.norm = nn.BatchNorm2d(in_channels)
                        
        self.encoder = self.make_encoder(self.en_cfgs[cfg], in_channels) 
        self.decoder = self.make_decoder(self.de_cfgs[cfg], self.initial_decoder_depth[cfg], out_channels)
        
        
    def latent(self,x):
        x = self.norm(x)
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        #x = self.encoder_fc(x)
        return x
        
    def forward(self, x):
        x = self.norm(x)
        x = self.encoder(x)

        shape2d = x.shape
        x = torch.flatten(x, 1)
        

        
        x = torch.reshape(x, shape2d)

        x = self.decoder(x)
        x = torch.sigmoid(x)

        return x
        
        
    def make_encoder(self, cfg, in_channels, batch_norm = True):
        layers = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
            
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    def make_decoder(self, cfg, in_channels, out_channels):
        layers = []
        
        for v in cfg:
            if v == "U":
                layers += [nn.Upsample(scale_factor = 2)]
            else:
                tconv2d = nn.ConvTranspose2d(in_channels,v,3,padding=1, stride=1)
                layers += [tconv2d, nn.ReLU(inplace=True)]
                in_channels = v
                
        layers += [nn.ConvTranspose2d(in_channels,out_channels,3,padding=1, stride=1)]
        
        return nn.Sequential(*layers)
    
    

class GolgiVGG(nn.Module):
    
    cfgs = {
        'A': [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        'B': [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M", 512, "M"],
        'D': [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
        'E': [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
    }
    
    
    def __init__(self,
                cfg = "B",
                dimensions = 196,
                in_channels = 5,
                num_classes = 2,
                 
                ):
        
        super(GolgiVGG, self).__init__()
        
        self.norm = nn.BatchNorm2d(in_channels)
        
        #self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.features = self.make_layers(self.cfgs[cfg], in_channels)
        
        self.classifier_1 = nn.Sequential(
            nn.Linear(512 * 2 * 2, 2048),
        )
        
        self.classifier_2 = nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
        )
            
        self.classifier_3 = nn.Sequential( 
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )
        
        
        

    def forward(self, x):
        x = self.norm(x)
        x = self.features(x)
        #x = self.avgpool(x)

        x = torch.flatten(x, 1)
        
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        x = self.classifier_3(x)

        x = self.softmax(x)
        return x
    
    def encoder(self, x):
        x = self.norm(x)
        x = self.features(x)
        return torch.flatten(x, 1)
    
    def encoder_c1(self, x):
        x = self.norm(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier_1(x)
        return x
    
    def encoder_c2(self, x):
        x = self.norm(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        return x
    
    def encoder_c3(self, x):
        x = self.norm(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier_1(x)
        x = self.classifier_2(x)
        x = self.classifier_3(x)
        return x

    def make_layers(self, cfg, in_channels, batch_norm = True):
        layers = []
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
            
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    

        
    def vgg(cfg, in_channels,  **kwargs):
        model = GolgiVGG(make_layers(cfgs[cfg], in_channels), **kwargs)
        return model
    
    
class GolgiVAE(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 latent_dim,
                 hidden_dims = None, **kwargs):
        
        super(GolgiVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512, 512, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= out_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 1, 1)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]
    
    def loss_function(self,target,output,
                      *args, 
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = output[0]
        mu = output[1]
        log_var = output[2]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, target)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
