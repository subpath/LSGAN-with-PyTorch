import torch
import torch.nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
import toml

from model import Generator, Discriminator, pack, write_loss_plot, save_images, save_model

CONFIG = toml.load('config.toml')['hyperparameters']


class LSGAN:

    def __init__(self, CONFIG):
        self.batch_size = CONFIG['batch_size']
        self.latent_input = CONFIG['latent_input']
        self.nb_image_to_gen = CONFIG['nb_image_to_gen']
        self.image_size = CONFIG['image_size']
        self.image_channels = CONFIG['image_channels']
        self.save_path = CONFIG['save_path']
        self.packing = CONFIG['packing']
        self.real_label_smoothing = bool(CONFIG['real_label_smoothing'])
        self.fake_label_smoothing = bool(CONFIG['fake_label_smoothing'])
        self.nb_discriminator_step = CONFIG['nb_discriminator_step']
        self.nb_generator_step = CONFIG['nb_generator_step']

        # Device (cpu or gpu)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Models
        self.generator = Generator(CONFIG['latent_input'], CONFIG['model_complexity'], CONFIG['dropout_prob'],
                                   CONFIG['weights_mean'], CONFIG['weights_std'],
                                   CONFIG['image_channels']).to(self.device)
        self.discriminator = Discriminator(CONFIG['model_complexity'], CONFIG['weights_mean'], CONFIG['weights_std'],
                                           CONFIG['packing'], CONFIG['image_channels']).to(self.device)

        print("------- GENERATOR ---------")
        print(self.generator)
        print("------- DISCRIMINATOR ---------")
        print(self.discriminator)

        # Optimizers
        self.D_optimiser = optim.Adam(self.discriminator.parameters(), lr=CONFIG['learning_rate'],
                                      betas=(CONFIG['beta1'], CONFIG['beta2']))
        self.G_optimiser = optim.Adam(self.generator.parameters(), lr=CONFIG['learning_rate'],
                                      betas=(CONFIG['beta1'], CONFIG['beta2']))

        self.generator_losses = []
        self.discriminator_losses = []

        self.saved_latent_input = torch.randn(
            (CONFIG['nb_image_to_gen'] * CONFIG['nb_image_to_gen'], CONFIG['latent_input'], 1, 1)).to(self.device)

        # Create directory for the results if it doesn't already exists
        import os
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.save_path + "real/", exist_ok=True)

    def load_dataset(self):
        image_size = 32
        batch_size = 128
        root = "../datasets/MNIST_data"
        trans = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        # Load dataset
        train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)

        print('Number of images: ', len(train_set))
        print('Sample image shape: ', train_set[0][0].shape, end='\n\n')

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True)

    def train(self, nb_epoch=CONFIG['nb_epoch']):
        print("Start training.")

        for epoch in range(nb_epoch):

            print("Epoch : " + str(epoch))
            g_loss = []
            d_loss = []

            for batch_id, (x, target) in enumerate(self.train_loader):
                real_batch_data = x.to(self.device)
                current_batch_size = x.shape[0]

                packed_real_data = pack(real_batch_data, self.packing)
                packed_batch_size = packed_real_data.shape[0]

                # labels
                label_real = torch.full((packed_batch_size,), 1, device=self.device).squeeze()
                label_fake = torch.full((packed_batch_size,), 0, device=self.device).squeeze()
                # smoothed real labels between 0.7 and 1, and fake between 0 and 0.3
                label_real_smooth = torch.rand((packed_batch_size,)).to(self.device).squeeze() * 0.3 + 0.7
                label_fake_smooth = torch.rand((packed_batch_size,)).to(self.device).squeeze() * 0.3

                temp_discriminator_loss = []
                temp_generator_loss = []

                ### Train discriminator multiple times
                for i in range(self.nb_discriminator_step):
                    loss_discriminator_total = self.train_discriminator(packed_real_data,
                                                                        current_batch_size,
                                                                        label_real_smooth if self.real_label_smoothing else label_real,
                                                                        label_fake_smooth if self.fake_label_smoothing else label_fake)

                    temp_discriminator_loss.append(loss_discriminator_total.item())
                    # print("Discriminator step ", str(i), " with loss : ", loss_discriminator_total.item())

                ### Train generator multiple times
                for i in range(self.nb_generator_step):
                    loss_generator_total = self.train_generator(current_batch_size, label_real)
                    temp_generator_loss.append(loss_generator_total.item())

                if batch_id == len(self.train_loader) - 2:
                    save_images(real_batch_data, self.save_path + "real/", self.image_size, self.image_channels,
                                self.nb_image_to_gen, epoch)

                ### Keep track of losses
                d_loss.append(torch.mean(torch.tensor(temp_discriminator_loss)))
                g_loss.append(torch.mean(torch.tensor(temp_generator_loss)))

            self.discriminator_losses.append(torch.mean(torch.tensor(d_loss)))
            self.generator_losses.append(torch.mean(torch.tensor(g_loss)))

            save_images(self.generator(self.saved_latent_input), self.save_path + "gen_", self.image_size,
                        self.image_channels, self.nb_image_to_gen, epoch)

            write_loss_plot(self.generator_losses, "G loss", self.save_path, clear_plot=False)
            write_loss_plot(self.discriminator_losses, "D loss", self.save_path, clear_plot=True)

        print("Training finished.")

    def train_discriminator(self, real_data, current_batch_size, real_label, fake_label):

        # Generate with noise
        latent_noise = torch.randn(current_batch_size, self.latent_input, 1, 1, device=self.device)
        generated_batch = self.generator(latent_noise)
        fake_data = pack(generated_batch, self.packing)

        ### Train discriminator
        self.discriminator.zero_grad()

        # Train on real data
        real_prediction = self.discriminator(real_data).squeeze()
        loss_discriminator_real = self.discriminator.loss(real_prediction, real_label)
        # loss_discriminator_real.backward()

        # Train on fake data
        fake_prediction = self.discriminator(fake_data.detach()).squeeze()
        loss_discriminator_fake = self.discriminator.loss(fake_prediction, fake_label)
        # loss_discriminator_fake.backward()

        # Add losses
        loss_discriminator_total = loss_discriminator_real + loss_discriminator_fake
        loss_discriminator_total.backward()
        self.D_optimiser.step()
        return loss_discriminator_total

    def train_generator(self, current_batch_size, real_label):

        # Generate with noise
        latent_noise = torch.randn(current_batch_size, self.latent_input, 1, 1, device=self.device)
        generated_batch = self.generator(latent_noise)
        fake_data = pack(generated_batch, self.packing)

        ### Train generator
        self.generator.zero_grad()

        fake_prediction = self.discriminator(fake_data).squeeze()

        # Loss
        loss_generator = self.generator.loss(fake_prediction, real_label)
        loss_generator.backward()
        self.G_optimiser.step()

        return loss_generator

    def save_models(self):
        save_model(self.generator, self.save_path, "generator_end")
        save_model(self.discriminator, self.save_path, "discriminator_end")


if __name__ == '__main__':
    # Create trainer for the LSGAN
    LSGAN = LSGAN(CONFIG)

    # Load the dataset
    LSGAN.load_dataset()

    # Start the training process
    LSGAN.train()

    # Save models
    LSGAN.save_models()
