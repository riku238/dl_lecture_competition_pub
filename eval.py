import os
import numpy as np
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed

def collate_fn_test(batch):
    X_batch = torch.stack([item[0].float() for item in batch])
    subject_idxs_batch = torch.tensor([item[1] for item in batch])
    return X_batch, subject_idxs_batch

@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    savedir = os.path.dirname(args.model_path)
    
    # ------------------
    #    Dataloader
    # ------------------    
    test_set = ThingsMEGDataset(
        split="test",
        data_dir=args.data_dir,
        resample_rate=args.resample_rate,
        filter_params=args.filter_params,
        scaling=args.scaling,
        baseline_correction=args.baseline_correction
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        shuffle=False, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        collate_fn=collate_fn_test
    )

    # ------------------
    #       Model
    # ------------------
    model = BasicConvClassifier(
        num_classes=test_set.num_classes, 
        seq_len=test_set.seq_len, 
        in_channels=test_set.num_channels
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # ------------------
    #  Start evaluation
    # ------------------ 
    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device).float()).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(savedir, "submission.npy"), preds)
    cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")

if __name__ == "__main__":
    run()
