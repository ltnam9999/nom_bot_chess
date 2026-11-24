"""
nom_bot_chess - Standalone Training Script
===========================================
Train the ChessResNet model locally or on a server via command line:
    python train.py --max-games 25000 --epochs 8 --batch-size 256
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import numpy as np
from tqdm import tqdm

import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "play")))
from uci_engine import encode_board, move_to_index, ChessResNet

DEFAULT_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "best_chess_bot.pt")

DATASET_URL = "https://database.nikonoel.fr/lichess_elite_2020-08.zip"
ZIP_PATH = "lichess_elite_2020-08.zip"
PGN_PATH = "lichess_elite_2020-08.pgn"

class ChessDataset(Dataset):
    def __init__(self, x, p, v):
        self.x = torch.from_numpy(x)
        self.p = torch.from_numpy(p)
        self.v = torch.from_numpy(v).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.p[idx], self.v[idx]

def download_and_extract():
    if not os.path.exists(PGN_PATH):
        if not os.path.exists(ZIP_PATH):
            print(f"Downloading {DATASET_URL} (~103 MB)...")
            urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
            print("Download completed!")
        print("Extracting PGN archive...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        print(f"PGN ready: {PGN_PATH} ({os.path.getsize(PGN_PATH)/1e6:.1f} MB)")

def prepare_dataset(max_games=25000, sample_freq=2, cache_file=None):
    if cache_file is None:
        cache_file = f"dataset_split_cache_{max_games}.npz"

    if os.path.exists(cache_file):
        print(f"Loading cached dataset from '{cache_file}'...")
        data = np.load(cache_file)
        return (data['train_x'], data['train_p'], data['train_v'],
                data['val_x'], data['val_p'], data['val_v'],
                data['test_x'], data['test_p'], data['test_v'])

    download_and_extract()
    print(f"Processing {max_games} games with 90/5/5 Game-Level split...")

    train_x, train_p, train_v = [], [], []
    val_x, val_p, val_v = [], [], []
    test_x, test_p, test_v = [], [], []

    with open(PGN_PATH, "r", encoding="utf-8", errors="ignore") as f:
        game_idx = 0
        pbar = tqdm(total=max_games, desc="Parsing PGN")
        while game_idx < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            result = game.headers.get("Result", "*")
            if result == "1-0": white_val = 1.0
            elif result == "0-1": white_val = -1.0
            elif result == "1/2-1/2": white_val = 0.0
            else: continue

            split_bin = game_idx % 20
            target = "train" if split_bin < 18 else ("val" if split_bin == 18 else "test")

            board = game.board()
            move_count = 0
            for move in game.mainline_moves():
                move_count += 1
                if move_count > 6 and (move_count % sample_freq == 0):
                    turn = board.turn
                    encoded = encode_board(board)
                    m_idx = move_to_index(move, turn)
                    pos_val = white_val if turn else -white_val

                    if target == "train":
                        train_x.append(encoded); train_p.append(m_idx); train_v.append(pos_val)
                    elif target == "val":
                        val_x.append(encoded); val_p.append(m_idx); val_v.append(pos_val)
                    else:
                        test_x.append(encoded); test_p.append(m_idx); test_v.append(pos_val)
                board.push(move)
            game_idx += 1
            pbar.update(1)
        pbar.close()

    print(f"Saving cache to '{cache_file}'...")
    np.savez_compressed(
        cache_file,
        train_x=np.array(train_x, dtype=np.float32), train_p=np.array(train_p, dtype=np.int64), train_v=np.array(train_v, dtype=np.float32),
        val_x=np.array(val_x, dtype=np.float32), val_p=np.array(val_p, dtype=np.int64), val_v=np.array(val_v, dtype=np.float32),
        test_x=np.array(test_x, dtype=np.float32), test_p=np.array(test_p, dtype=np.int64), test_v=np.array(test_v, dtype=np.float32)
    )
    return (np.array(train_x), np.array(train_p), np.array(train_v),
            np.array(val_x), np.array(val_p), np.array(val_v),
            np.array(test_x), np.array(test_p), np.array(test_v))

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using computing device: {device}")

    train_x, train_p, train_v, val_x, val_p, val_v, test_x, test_p, test_v = prepare_dataset(
        max_games=args.max_games, sample_freq=args.sample_freq
    )

    total_samples = len(train_x) + len(val_x) + len(test_x)
    print(f"Total positions: {total_samples:,} (Train: {len(train_x):,}, Val: {len(val_x):,}, Test: {len(test_x):,})")

    train_loader = DataLoader(ChessDataset(train_x, train_p, train_v), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(ChessDataset(val_x, val_p, val_v), batch_size=args.batch_size, shuffle=False)

    model = ChessResNet().to(device)
    crit_policy = nn.CrossEntropyLoss()
    crit_value = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        r_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs} [Train]")
        for bx, bp, bv in pbar:
            bx, bp, bv = bx.to(device), bp.to(device), bv.to(device)
            optimizer.zero_grad()
            p_out, v_out = model(bx)
            loss = crit_policy(p_out, bp) + 0.5 * crit_value(v_out, bv)
            loss.backward()
            optimizer.step()

            r_loss += loss.item() * bx.size(0)
            correct += (p_out.argmax(1) == bp).sum().item()
            total += bx.size(0)
            pbar.set_postfix({"loss": f"{r_loss/total:.4f}", "acc": f"{correct/total*100:.2f}%"})

        scheduler.step()
        train_loss, train_acc = r_loss / total, correct / total * 100

        # Validation
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for bx, bp, bv in val_loader:
                bx, bp, bv = bx.to(device), bp.to(device), bv.to(device)
                p_out, v_out = model(bx)
                loss = crit_policy(p_out, bp) + 0.5 * crit_value(v_out, bv)
                v_loss += loss.item() * bx.size(0)
                v_corr += (p_out.argmax(1) == bp).sum().item()
                v_tot += bx.size(0)

        val_loss, val_acc = v_loss / v_tot, v_corr / v_tot * 100
        print(f"Epoch {epoch:02d} Summary: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  ⭐ Saved best checkpoint to '{args.save_path}' (Val Loss: {val_loss:.4f})")

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train nom_bot_chess")
    parser.add_argument("--max-games", type=int, default=25000, help="Number of games to parse")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sample-freq", type=int, default=2, help="Sample frequency (plies)")
    parser.add_argument("--save-path", type=str, default=DEFAULT_WEIGHTS_PATH, help="Path to save best weights")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()
    train(args)
