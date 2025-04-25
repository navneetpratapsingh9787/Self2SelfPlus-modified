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
    
    # Create Model Instance
    model = models.Self2SelfPlus(opt)
    model = models.assignOnGpu(opt, model)
    
    # Create Optimizer Instance
    optimizer = optim.Adam(model.net.parameters(), 
                          lr=opt.lr, 
                          betas=(opt.beta1, opt.beta2))

    for data in dataLoader:
        # Load Data
        noisyImage, name = data["noisyImage"], data["name"]
        
        # Assign Device
        noisyImage = models.preprocessData(opt, noisyImage)
        
        # Initialize Best Result Tracking
        best_loss = float('inf')  # Track minimum total loss
        best_denoised_image = None
        
        print("================================================================================================================================")
        print(f"< Image File Name : {name[0]} >")
    
        with tqdm(total=opt.numIters) as pBar:
            for iter in range(1, opt.numIters + 1):
                # Train Denoising Autoencoder
                optimizer.zero_grad()
                loss = model(noisyImage, mode="train")
                
                # Compute Loss Components
                lossSS = loss["self_supervised"].item()
                lossIQA = loss["iqa"].item()
                lossPerceptual = loss["perceptual"].item()
                
                # Compute Total Loss
                total_loss = sum(loss.values()).mean()
                
                # Back-Propagation
                total_loss.backward()
                optimizer.step()

                # Perform Inference Every 100 Iterations or at the End
                if iter % 100 == 0 or iter == opt.numIters:
                    with torch.no_grad():
                        denoisedImage = model(noisyImage, "inference")
                        denoisedImage = denoisedImage.clamp(0, 1)
                        
                        # Recompute Loss for Inference Output
                        inference_loss = model(noisyImage, mode="train")
                        inference_total_loss = sum(inference_loss.values()).mean().item()
                        
                        # Update Best Result
                        if inference_total_loss < best_loss:
                            best_loss = inference_total_loss
                            best_denoised_image = denoisedImage
                            print(f"New best total loss: {best_loss:.8f} at iteration {iter}")

                # Show Training Procedure
                pBar.set_description(desc=f"[{iter}/{opt.numIters}] < Loss(Self-Supervised):{lossSS:.8f} | Loss(IQA):{lossIQA:.8f} | Loss(Perceptual):{lossPerceptual:.8f} | Total Loss: {total_loss.item():.8f} >")
                pBar.update()

        # Save Best Denoised Image
        if best_denoised_image is not None:
            utils.saveImage(best_denoised_image, imageSavePath, name[0])
            print(f"Saved best denoised image with total loss: {best_loss:.8f}")
        else:
            print("No denoised image saved (inference not performed).")
        print()

if __name__ == "__main__":
    main()
