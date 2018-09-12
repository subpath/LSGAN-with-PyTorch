import torch
import torch.nn as nn
from torchvision.utils import save_image
from torchvision.utils import make_grid
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def weights_init_general(model, mean, std):
    for m in model._modules:
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            model._modules[m].weight.data.normal_(mean, std)
            model._modules[m].bias.data.zero_()


class Generator(nn.Module):
    def __init__(self, input_size, general_complexity, dropout_prob, weights_mean, weights_std, image_channels):
        super(Generator, self).__init__()

        self.loss = nn.MSELoss()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(input_size, 4 * general_complexity, 4, 1, 0, bias=False),
            nn.BatchNorm2d(4 * general_complexity),
            nn.ReLU(True),
            nn.Dropout2d(dropout_prob)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(4 * general_complexity, 2 * general_complexity, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2 * general_complexity),
            nn.ReLU(True),
            nn.Dropout2d(dropout_prob)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(2 * general_complexity, 1 * general_complexity, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1 * general_complexity),
            nn.ReLU(True),
            nn.Dropout2d(dropout_prob)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(1 * general_complexity, 1 * image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.all_layers = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4)

        weights_init_general(self, weights_mean, weights_std)

    def forward(self, input):
        output = self.all_layers(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, general_complexity, weights_mean, weights_std, packing, image_channels):
        super(Discriminator, self).__init__()

        self.loss = nn.MSELoss()

        self.layer1 = nn.Sequential(
            nn.Conv2d(image_channels * packing, general_complexity, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(general_complexity, general_complexity * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(general_complexity * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(general_complexity * 2, general_complexity * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(general_complexity * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(general_complexity * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.all_layers = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        )

        weights_init_general(self, weights_mean, weights_std)

    def forward(self, input):
        output = self.all_layers(input)
        return output


def pack(input, packing):
    # Number of elements that need to be added to the input tensor
    nb_to_add = (packing - (input.shape[0] % packing)) % packing

    # Add elements to the input if not a round number for the packing number
    if nb_to_add > 0:
        input = torch.cat((input, input[-nb_to_add:].view(nb_to_add, 3, input.shape[2], input.shape[3])))

    # Reshape the tensor so it is packed
    packed_output = input.view(-1, input.shape[1] * packing, input.shape[2], input.shape[3])
    return packed_output


# Initialise weights of the model with certain mean and standard deviation
def weights_init_general(model, mean, std):
    for m in model._modules:
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            model._modules[m].weight.data.normal_(mean, std)
            model._modules[m].bias.data.zero_()


def write_loss_plot(loss, loss_label, save_path, clear_plot=True):
    plt.plot(loss, label=loss_label)
    plt.legend(loc="best")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(save_path + "losses.png")

    if clear_plot:
        plt.clf()


def save_model(model, save_path, name):
    print("Saving " + name + " to : " + save_path)
    torch.save(model.state_dict(), save_path + name + ".pt")



def rescale_for_rgb_plot(images):
    min_val = images.data.min()
    max_val = images.data.max()
    return (images.data - min_val) / (max_val - min_val)


def save_images(data, save_path, image_size, image_channels, num_row, epoch):
    image_list = []
    for i in range(len(data)):
        image_data = data[i].view(image_channels, image_size, image_size)
        image_data = rescale_for_rgb_plot(image_data)
        image_list.append(image_data)
    save_image(make_grid(image_list, nrow=num_row), save_path + "epoch_" + str(epoch) + ".png")
