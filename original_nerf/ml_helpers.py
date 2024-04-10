from tqdm import tqdm
from rendering import rendering
import torch

def training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, data_loader, save_path, type, device='cpu'):
    
    training_loss = []
    for epoch in (range(nb_epochs)):
        for batch in tqdm(data_loader):
            o = batch[:, :3].to(device)
            d = batch[:, 3:6].to(device)
            
            target = batch[:, 6:].to(device)
            
            prediction = rendering(model, o, d, tn, tf, nb_bins=nb_bins, device=device)
            
            loss = ((prediction - target)**2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
            
        scheduler.step()

        #torch.save(model.cpu(), f'model_nerf-epoch_{epoch}')
        if type == 'warmup':
            torch.save(model.cpu(), f'{save_path}/results/model_nerf_warmup')
        if type == 'epoch':
            torch.save(model.cpu(), f'{save_path}/results/model_nerf_epoch{epoch}')
        model.to(device)
        
    return training_loss