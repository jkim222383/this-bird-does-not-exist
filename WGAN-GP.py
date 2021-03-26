import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms, datasets, utils
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import pickle

# Weight initialization as suggested in literature
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def convtrans_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),    # bias not needed because of BatchNorm
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Generator(nn.Module):
    def __init__(self, LATENT_SIZE):
        super().__init__()
        self.main = nn.Sequential(
            convtrans_block(LATENT_SIZE, 512, kernel_size=4, stride=4, padding=0),
            convtrans_block(512, 256, kernel_size=4, stride=4, padding=1),
            convtrans_block(256, 128, kernel_size=4, stride=2, padding=1),
            convtrans_block(128, 64, kernel_size=4, stride=2, padding=1),
            convtrans_block(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

def conv_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),    # bias not needed because of BatchNorm
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            conv_block(3, 64, kernel_size=4, stride=2, padding=0),
            conv_block(64, 128, kernel_size=4, stride=2, padding=0),
            conv_block(128, 256, kernel_size=4, stride=2, padding=0),
            conv_block(256, 256, kernel_size=4, stride=2, padding=0),
            conv_block(256, 256, kernel_size=4, stride=2, padding=0),
            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=0),
        )

    def forward(self, x):
        return self.main(x)
    
# TODO: break this into multiple functions  
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATA_DIR = "data/consolidated/"
    BATCH_SIZE = 64
    SAMPLE_SIZE = 64

    # Images are 3 x 224 x 224
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=DATA_DIR,
                                transform=data_transforms)

    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            drop_last=False)

    # Plot samples
    sample_batch = next(iter(data_loader))
    plt.figure(figsize=(10, 8)); plt.axis("off"); plt.title("Sample Real Images")
    plt.imshow(np.transpose(utils.make_grid(sample_batch[0][:SAMPLE_SIZE], padding=1, normalize=True), (1, 2, 0)));

    LATENT_SIZE = 128
        
    generator = Generator(LATENT_SIZE)
    generator.apply(weights_init)

    critic = Critic()
    critic.apply(weights_init)

    # Change these lines when training from saved file (0 if from scratch)
    SAVED_EPOCH = 0
    FROM_SAVED = True

    PREFIX = "models/WGAN-GP/WGAN-GP_"
    IMG_PATH = PREFIX + "img_list.pickle"
    GEN_LOSSES_PATH = PREFIX + "gen_losses.pickle"
    CRI_LOSSES_PATH = PREFIX + "cri_losses.pickle"
    GIF_PATH = PREFIX + "images.gif"
    INIT_WEIGHTS_PATH = PREFIX + "initialization_"
    FIXED_NOISE_PATH = PREFIX + "fixed_noise.pt"

    if FROM_SAVED:
        GEN_PATH = PREFIX + f"gen_epoch{SAVED_EPOCH:0=3d}.pt"
        CRI_PATH = PREFIX + f"cri_epoch{SAVED_EPOCH:0=3d}.pt"

        generator.load_state_dict(torch.load(GEN_PATH))
        critic.load_state_dict(torch.load(CRI_PATH))
        fixed_noise = torch.load(FIXED_NOISE_PATH)

        with open(IMG_PATH, 'rb') as f:
            img_list = pickle.load(f)
        with open(GEN_LOSSES_PATH, 'rb') as f:
            gen_losses = pickle.load(f)
        with open(CRI_LOSSES_PATH, 'rb') as f:
            cri_losses = pickle.load(f)

        ITERS = SAVED_EPOCH * 590
    else:
        SAVED_EPOCH = 0
        ITERS = 0
        img_list = []
        gen_losses = []
        cri_losses = []
        
        try:
            # Comment out fail() when loading saved noise/init weights (Useful when comparing between architectures)
            fail()
            fixed_noise = torch.load(FIXED_NOISE_PATH)
            generator.load_state_dict(torch.load(INIT_WEIGHTS_PATH + "gen.pt"))
            critic.load_state_dict(torch.load(INIT_WEIGHTS_PATH + "cri.pt"))
            print("Previous fixed noise and initialization for gen/cri loaded!")
        except:      
            # Fixed noise vector we'll use to track image generation evolution
            fixed_noise = torch.randn(SAMPLE_SIZE, LATENT_SIZE, 1, 1, device=device)
            torch.save(fixed_noise, FIXED_NOISE_PATH)
            
            generator = Generator(LATENT_SIZE)
            generator.apply(weights_init)
            torch.save(generator.state_dict(), INIT_WEIGHTS_PATH + "gen.pt")

            critic = Critic()
            critic.apply(weights_init)
            torch.save(critic.state_dict(), INIT_WEIGHTS_PATH + "cri.pt")
            print("New fixed noise and initialization for gen/cri generated & saved!")

    generator.to(device)
    critic.to(device)

    img_list = []

    NUM_EPOCHS = 50

    # Original WGAN-GP proposes having 5 critic updates and 1 generator update every iteration
    # Here, we'll try more generator updates in case critic overpowers generator at later epochs
    n_cri = 1
    n_gen = 5
    λ = 10
    LR = 0.0002

    optimizerG = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizerC= optim.Adam(critic.parameters(), lr=LR, betas=(0.5, 0.999))

    print("Begin training...")

    for epoch in range(SAVED_EPOCH + 1, SAVED_EPOCH + NUM_EPOCHS + 1):
        for real_batch, _ in data_loader:
            loss_cri_iter = 0
            loss_gen_iter = 0
            C_real_iter = 0
            C_fake_iter = 0
            
            for _ in range(n_cri):
                optimizerC.zero_grad()

                # Train with real data
                real_batch = real_batch.to(device)
                real_labels = torch.ones((real_batch.shape[0],), dtype=torch.float, device=device)
                output_real = critic(real_batch).view(-1)
                C_real = output_real.mean()
            
                # Train with fake data
                noise = torch.randn(real_batch.shape[0], LATENT_SIZE, 1, 1, device=device)
                fake_batch = generator(noise)
                fake_labels = torch.zeros_like(real_labels)           
                output_fake = critic(fake_batch.detach()).view(-1)  
                C_fake = output_fake.mean()
            
                # Gradient Penalty
                ϵ = torch.rand(real_batch.shape[0], 1, 1, 1, device=device, requires_grad=True)
                interpolated = real_batch * ϵ + fake_batch.detach() * (1 - ϵ)
                output_interpolated = critic(interpolated)
                gradient = torch.autograd.grad(inputs=interpolated,
                                            outputs=output_interpolated,
                                            grad_outputs=torch.ones_like(output_interpolated), 
                                            create_graph=True)[0]

                gradient = gradient.view(len(gradient), -1)
                grad_norm = gradient.norm(2, dim=1)
                gp = torch.mean((grad_norm - 1) ** 2)            
                
                # Update critic weights and store loss
                loss_cri = C_fake - C_real + λ * gp
                loss_cri.backward()
                optimizerC.step()
                C_real_iter += C_real.item()
                C_fake_iter += C_fake.item()
                loss_cri_iter += loss_cri.item()
            
            for _ in range(n_gen):
                optimizerG.zero_grad()
                noise = torch.randn(real_batch.shape[0], LATENT_SIZE, 1, 1, device=device)
                fake_batch = generator(noise)
                output_fake = critic(fake_batch).view(-1)
                loss_gen = -output_fake.mean()
            
                # Update generator weights and store loss
                loss_gen.backward()
                optimizerG.step()
                loss_gen_iter += loss_gen.item()
            
            C_real_iter /= n_cri
            C_fake_iter /= n_cri
            loss_cri_iter /= n_cri
            loss_gen_iter /= n_gen
            
            # Track losses for later use
            gen_losses.append(loss_gen_iter)
            cri_losses.append(loss_cri_iter)
            
            if (ITERS % 50 == 0):
                print(f"Epoch ({epoch}/{SAVED_EPOCH + NUM_EPOCHS})",
                    f"Iteration ({ITERS})",
                    f"Loss_G: {loss_gen_iter:.4f}",
                    f"Loss_C: {loss_cri_iter:.4f}",
                    f"C_real: {C_real:.4f}",  
                    f"C_fake: {C_fake:.4f}")
            ITERS += 1
        
        ### Store loss and track image evolution (tracking image every 2 epochs to reduce memory usage)
        if epoch % 2 == 0:
            with torch.no_grad():
                fake_images = generator(fixed_noise).detach().cpu()
            img_list.append(utils.make_grid(fake_images, nrow=8, normalize=True))
            
            # Display generated images during training
            plt.figure(figsize=(8, 8)); plt.axis("off"); plt.title(f"Epoch {epoch} Generated Images")
            plt.imshow(np.transpose(img_list[-1], (1,2,0)));
            plt.show()
        
        # Save weights every 5 epochs
        if epoch % 5 == 0:
            torch.save(generator.state_dict(), PREFIX + f"gen_epoch{epoch:0=3d}.pt")
            torch.save(critic.state_dict(), PREFIX + f"cri_epoch{epoch:0=3d}.pt")  
            
    print("Finished training!")