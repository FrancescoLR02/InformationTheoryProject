from .LIBRARIES_FUNCTIONS import *

def ShowSomeImages(model, testDataset, device, howmany=5):

   model.eval()
   fig, axs = plt.subplots(howmany, 2, figsize=(4, howmany*2))

   for i in range(howmany):
    img, label = random.choice(testDataset)

    x = img.unsqueeze(0).to(device)

    with torch.no_grad():
         recon, _, _, _, _ = model(x, label)

    original = img.cpu().squeeze().numpy()
    reconstructed = recon.cpu().squeeze().numpy().reshape(28, 28)

    axs[i, 0].imshow(original, cmap="gist_gray")
    axs[i, 0].set_title("Original")
    axs[i, 0].set_xticks([])
    axs[i, 0].set_yticks([])

    axs[i, 1].imshow(reconstructed, cmap="gist_gray")
    axs[i, 1].set_title("Reconstruction")
    axs[i, 1].set_xticks([])
    axs[i, 1].set_yticks([])

   plt.tight_layout()
   plt.show()