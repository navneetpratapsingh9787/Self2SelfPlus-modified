from torch import optim
from tqdm import tqdm
import config
from data import dataloaders
from models import models
from utils import utils

def main():
    # Read Options
    opt = config.readArguments()
    
    # Create DataLoader Instances
    imageSavePath, dataLoader = dataloaders.getDataLoaders(opt)
    
    # Create Save Directory
    utils.mkdirs(imageSavePath)
    
    # Create Model Instance (Moved outside loop)
    model = models.Self2SelfPlus(opt)
    model = models.assignOnGpu(opt, model)
    
    # Create Optimizer Instance (Moved outside loop)
    optimizer = optim.Adam(model.net.parameters(), 
                          lr=opt.lr, 
                          betas=(opt.beta1, opt.beta2))

    for data in dataLoader:
        # Load Data
        noisyImage, name = data["noisyImage"], data["name"]
        
        # Assign Device
        noisyImage = models.preprocessData(opt, noisyImage)
        
        # Get Image File Name
        print("================================================================================================================================")
        print(f"< Image File Name : {name[0]} >")
    
        with tqdm(total=opt.numIters) as pBar:
            for iter in range(1, opt.numIters + 1):
                # Train Denoising Autoencoder
                optimizer.zero_grad()
                loss = model(noisyImage, mode="train")
                
                # Compute Loss Components
                lossSS = loss["Self-Supervised"].item()
                lossIQA = loss["IQA"].item()
                lossPerceptual = loss["Perceptual"].item()  # Added
                
                # Back-Propagation
                total_loss = sum(loss.values()).mean()
                total_loss.backward()
                optimizer.step()

                # Show Training Procedure
                pBar.set_description(desc=f"[{iter}/{opt.numIters}] < Loss(Self-Supervised):{lossSS:.8f} | Loss(IQA):{lossIQA:.8f} | Loss(Perceptual):{lossPerceptual:.8f} >")
                pBar.update()

        # Get Inference Result
        denoisedImage = model(noisyImage, "inference")
        
        # Clamp Image
        denoisedImage = denoisedImage.clamp(0, 1)
        
        # Save Image
        utils.saveImage(denoisedImage, imageSavePath, name[0])
        print()

if __name__ == "__main__":
    main()
