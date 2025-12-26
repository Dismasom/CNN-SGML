import argparse
import os
from network import UNET_CBAM, FNO2D
import torch
from train_set import ModeLoss, G_train,device, loadData
from torch import optim
from torch.nn import functional as F
import numpy as np

def parse_args():
    
    parser = argparse.ArgumentParser(description="Training Configuration")
    # Architecture and hyperparameters
    parser.add_argument("--ngf", type=int, default=32, help="UNet base channels")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--in_ch", type=int, default=1, help="input channels")
    parser.add_argument("--out_ch", type=int, default=1, help="output channels")
    parser.add_argument("--num_layers", type=int, default=5, help="UNet layers")
    parser.add_argument("--lr", type=float, default=4e-4, help="learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--epochs", type=int, default=20, help="epochs")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability in UNET_CBAM decoder")

    # Training mode
    parser.add_argument("--mode",type=str,default="PINN",help="Training mode: ML / SGML / PINN / FNO",)

    # Model saving
    parser.add_argument("--save_interval",type=int,default=10,help="Epoch interval for evaluation and model saving on validation set",)
    parser.add_argument("--top_k",type=int,default=5,help="Number of best models to keep based on validation error",)

    # Data and saving paths
    parser.add_argument("--data_root",type=str,default="./DATA",help="Data root directory containing TRAIN / VAL subdirectories",)
    parser.add_argument("--train_data", type=str, default="train.npy", help="Training set ")
    parser.add_argument("--val_data", type=str, default="val.npy", help="Validation")
    parser.add_argument("--save_root",type=str,default="./checkpoints",help="Root directory for saving models and logs",)
    parser.add_argument("--u_norm", type=float, default=0.03451,help="Scale factor applied to prediction when saving output", )
    return parser.parse_args()

def main():
    args = parse_args()

    ngf = args.ngf
    layers = args.num_layers
    batch_size = args.batch_size
    in_ch, out_ch = args.in_ch, args.out_ch
    lr = args.lr
    beta1, beta2 = args.beta1, args.beta2
    epochs = args.epochs
    save_interval = max(1, args.save_interval)
    top_k = max(1, args.top_k)
    mode = str(args.mode).upper()
    dropout = args.dropout

    save_path = args.save_root
    root = args.data_root
    data_train = args.train_data
    data_val = args.val_data
    U_norm = args.u_norm

    if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

    # Load data
    train_loader = loadData(root, data_train, batch_size, shuffle=True)
    val_loader = loadData(root, data_val, 1, shuffle=False)

    # Declare model
    if mode == "FNO":
        G = FNO2D(in_ch, out_ch).to(device)
    else:
        G = UNET_CBAM(in_ch, out_ch, ngf, layers, dropout=dropout).to(device)
    print("mode:", mode)
    

    # Loss function & optimizer
    Loss = ModeLoss().to(device)
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))

    # start training
    G.train()  # (distinguish from .eval)
    G_Loss = []  #  loss data for one epoch
    if mode == "PINN":
        data_loss, exterior_loss, physics_loss, boundary_loss = [], [], [], []
    best_models = []  # Initialize a list to store loss values and model paths
    
    for epoch in range(epochs):

        G_losses, batch, g_l = [], 0, 0  
        # -------- training phase (enable Dropout) --------
        G.train()
        for X, in train_loader:
            batch += 1
            loss, loss_dict = G_train(G, X, Loss, optimizer_G, epoch, epochs, mode)
            G_losses.append(loss)
            l_batch = loss
            # Print the average loss for each batch
            g_l = np.array(G_losses).mean()
            if mode == "PINN":
                print('[%d / %d]: batch#%d  loss= %.6f  data= %.6f  bnd= %.6f  ext= %.6f  phy= %.6f' %
                      (epoch + 1, epochs, batch,
                       l_batch,
                       loss_dict.get('data_loss', 0.0),
                       loss_dict.get('boundary_loss', 0.0),
                       loss_dict.get('exterior_loss', 0.0),
                       loss_dict.get('physics_loss', 0.0)))
                data_loss.append(loss_dict.get('data_loss', 0.0))
                exterior_loss.append(loss_dict.get('exterior_loss', 0.0))
                physics_loss.append(loss_dict.get('physics_loss', 0.0))
                boundary_loss.append(loss_dict.get('boundary_loss', 0.0))
            else:
                print('[%d / %d]: batch#%d  loss= %.6f ' %
                      (epoch + 1, epochs, batch,
                       l_batch,
                    ))

        # Evaluate on the validation set and try to save the model at specified intervals
        if (epoch + 1) % save_interval == 0:
            G.eval()
            ERROR = []
            U_norm = args.u_norm
            with torch.no_grad():
                for X, in val_loader:
                    X = X.to(torch.float).to(device)

                    approx = X[:, :, :, 0:256]
                    mask = X[:, :, :, 256:512]
                    target = X[:, :, :, 512:768]
  
                    if mode == "SGML":
                        x = approx
                    else:  
                        x = mask

                    pred = G(x)

                    # Use RMSE as the validation metric
                    rmse = torch.sqrt(F.mse_loss(pred*U_norm, target)).item()
                    ERROR.append(rmse)

            # Remove outliers
            ERROR = np.array(ERROR)
            ERROR = ERROR[ERROR < 5*np.median(ERROR)]  

            # top_k model management: only save models that enter the top_k and delete those that are pushed out
            current_path = os.path.join(save_path, 'generator_%d.pth' % (epoch + 1))
            best_models.append((ERROR.mean(), current_path))
            best_models.sort(key=lambda x: x[0])

            to_remove = []
            if len(best_models) > top_k:
                to_remove = best_models[top_k:]
                best_models = best_models[:top_k]

            if any(p == current_path for _, p in best_models):
                torch.save(G, current_path)

            for _, model_path in to_remove:
                if os.path.exists(model_path):
                    os.remove(model_path)

            print('epoch:', epoch + 1, 'RMSE:', ERROR.mean(), 'best_models:', best_models)
            G.train()

        G_Loss.append(g_l)
   
    # Save training results
    torch.save(G, os.path.join(save_path, 'generator.pth'))
    np.save(os.path.join(save_path, 'G_Loss.npy'), np.array(G_Loss))
    if mode == "PINN":
        np.save(os.path.join(save_path, 'data_loss.npy'), np.array(data_loss))
        np.save(os.path.join(save_path, 'physics_loss.npy'), np.array(physics_loss))
        np.save(os.path.join(save_path, 'boundary_loss.npy'), np.array(boundary_loss))
        np.save(os.path.join(save_path, 'exterior_loss.npy'), np.array(exterior_loss))

    print("Done!")

if __name__ == '__main__':
    main()
