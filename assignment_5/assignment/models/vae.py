import torch


class VanillaVAE(nn.Module):
    """
    Implementation of a fully connect autoencoder for
    denoising  MNIST images
    """

    def __init__(self, in_size=(1, 32, 32), sizes=[1024, 128, 10], act="ReLU"):
        """Model initlaizer"""
        assert np.prod(in_size) == sizes[0]
        super().__init__()

        self.in_size = in_size
        self.sizes = sizes
        self.activation = get_act(act)

        self.encoder = self._make_encoder()
        self.decoder = self._make_decoder()
        self.fc_mu = nn.Linear(sizes[-2], sizes[-1])
        self.fc_sigma = nn.Linear(sizes[-2], sizes[-1])
        return

    def _make_encoder(self):
        """Defining encoder"""
        layers = [nn.Flatten()]

        # adding fc+act+drop for each layer
        for i in range(len(self.sizes) - 2):
            layers.append(nn.Linear(in_features=self.sizes[i], out_features=self.sizes[i + 1]))
            layers.append(self.activation)

        # replacing last act and dropout with sigmoid
        encoder = nn.Sequential(*layers)
        return encoder

    def _make_decoder(self):
        """Defining decoder"""
        layers = []

        # adding fc+act+drop for each layer
        for i in range(1, len(self.sizes)):
            layers.append(nn.Linear(in_features=self.sizes[-i], out_features=self.sizes[-i - 1]))
            layers.append(self.activation)

        # replacing last act and dropout with sigmoid
        layers = layers[:-1] + [nn.Sigmoid()]
        decoder = nn.Sequential(*layers)
        return decoder

    def reparameterize(self, mu, log_var):
        """Reparametrization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # random sampling happens here
        z = mu + std * eps
        return z

    def forward(self, x):
        """Forward pass"""

        # encoding and computng statistics
        x_enc = self.encoder(x)
        mu = self.fc_mu(x_enc)
        log_var = self.fc_sigma(x_enc)

        # reparametrization trick
        z = self.reparameterize(mu, log_var)

        # decoding
        x_hat_flat = self.decoder(z)
        x_hat = x_hat_flat.view(-1, *self.in_size)

        return x_hat, (z, mu, log_var)


class VariationalAutoEncoder(nn.Module):

    def __init__(self, input_size: tuple(int), encoder_layers: list(dict), decoder_layers: list(dict), latent_dim: tuple(int)):
        super(VariationalAutoEncoder, self).__init__()

        self.input_size = input_size
        self.encoder = build_model(encoder_layers)
        self.decoder = build_model(decoder_layers)

        self.fc_mu = nn.Linear(latent_dim[0], latent_dim[1])
        self.fc_sigma = nn.Linear(latent_dim[0], latent_dim[1])

    def forward(self, x):
        z = self.encode(x)
        x_out = self.decode(z)
        x_out = x_out.view(-1, *self.input_size)
        return x_out

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x):
        self.encoder(x)

        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)

        z = self.reparametrize(mu, sigma)

        return z

    def decode(self, z):
        return self.decoder(z)


class VanillaVAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None, **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
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

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
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

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss.detach(), "KLD": -kld_loss.detach()}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
