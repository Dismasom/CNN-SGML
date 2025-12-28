import argparse
from pathlib import Path
import cv2
import numpy as np
import torch


def _save_png(array2d: np.ndarray, out_path: Path, vmax: float | None = None):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(3.33, 3.33))
        ax.imshow(array2d, cmap="jet", interpolation="None", vmax=vmax)
        ax.axis("off")
        fig.savefig(out_path.as_posix(), bbox_inches="tight", pad_inches=0, transparent=False)
        plt.close(fig)

def Approximation_solution(input,pade=2.5,vos=1e-3):
    input = np.array(input*255, dtype=np.uint8)
    cont, _ = cv2.findContours(input, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cont_num=_.shape[1]
    dis_img = np.zeros(input.shape)
    if cont_num<=1: # Single connected domain
        cont = max(cont, key=cv2.contourArea)
        for y in range(dis_img.shape[0]):
            for x in range(dis_img.shape[1]):
                # Use pointPolygonTest to check if each point is inside the contour
                point = (x, y)
                if cv2.pointPolygonTest(cont, point, False) >= 0:
                    dis_img[y, x] = cv2.pointPolygonTest(cont, point, True)
    else: # Multiple connected domains
        external_contours = cont[0]
        internal_contours = cont[1]
        for x in range(input.shape[1]):
            for y in range(input.shape[0]):
                point = (x, y)
                if cv2.pointPolygonTest(external_contours, point, False) >= 0 and cv2.pointPolygonTest(internal_contours, point, False) <= 0:
                    dis_img[y, x] = min(abs(cv2.pointPolygonTest(external_contours, point, True)), abs(cv2.pointPolygonTest(internal_contours, point, True)))
    dis_img=dis_img*0.0001
    d_max=np.max(dis_img)
    Appro_field = np.where(dis_img != 0, pade/vos/4*(d_max**2)*(1-((d_max-dis_img)**2)/(d_max**2)), 0)

    return Appro_field.astype(np.float32)


def main():
        parser = argparse.ArgumentParser(description="Run inference on merged TEST npy (N,H,W).")
        parser.add_argument("--data", default="./Data/test.npy", help="Merged test .npy file, shape (N,256,768)")
        parser.add_argument("--model", default="./Model/SGML.pth", help="Model .pth path (torch.save'd model)")
        parser.add_argument("--out",default="TEST",help="Output directory ",)
        parser.add_argument("--mode",type=str,default="SGML",help="SGML uses approx as input; ML/PINN uses mask as input",)
        parser.add_argument("--u_norm",type=float,default=0.03451,help="Scale factor applied to prediction when saving output",)
        parser.add_argument("--test_set", default=True )
        args = parser.parse_args()
        
        data_path = Path(args.data)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mode = str(args.mode).upper()

        # Read merged test set (mmap does not occupy entire memory block)
        arr = np.load(data_path, mmap_mode="r")
        n, h, w = arr.shape

        # Load model (only load once)

        model=torch.load(args.model)
        model.to(device).eval()

        print(f"data: {data_path}  shape={arr.shape}")
        print(f"model: {args.model}  device={device}")
        print(f"out: {out_dir}")

        for i in range(n):
                u_norm=args.u_norm
                if args.test_set:
                        sample = np.array(arr[i], copy=True).astype(np.float32, copy=False)  # (256,768)
                        x = sample[None, None, :, :]  # (1,1,H,W)
                        X = torch.from_numpy(x).to(device)

                        approx = X[:, :, :, 0:256]
                        mask = X[:, :, :, 256:512]
                        target = X[:, :, :, 512:768]

                        if mode == "SGML" :
                            inp = approx
                            with torch.no_grad():
                                pred = model(inp / u_norm)
                        else:
                            inp= mask
                            with torch.no_grad():
                                pred = model(inp)

                pred_np = pred[0, 0].detach().cpu().numpy()  # (256,256)
                pred_np = pred_np*u_norm
                inp_np = inp[0, 0].detach().cpu().numpy()
                tgt_np = target[0, 0].detach().cpu().numpy()

                save_data = np.concatenate((inp_np, pred_np, tgt_np), axis=1)
                np.save(out_dir / f"{i}.npy", save_data)

                # Visualization: Save concatenated and predicted images
                vmax_all = float(np.max(save_data[:, 512:]))
                _save_png(save_data, out_dir / f"output_all_{i}.png", vmax=vmax_all)

                if (i + 1) % 50 == 0 or i == n - 1:
                        print(f"[{i+1}/{n}] done")


if __name__ == "__main__":
        main()






