# ♟️ nom_bot_chess (Elo 1500+)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Lichess](https://img.shields.io/badge/Lichess-Bot%20API-orange.svg)](https://lichess.org)

**nom_bot_chess** is an AI chess engine operating at an estimated **Elo 1500 – 1800+**, built upon a 10.3M-parameter Deep Residual Neural Network (ResNet). It was trained via supervised learning on approximately 1 million master-level positions from the Lichess Elite Database (August 2020 archive) and integrated with Minimax Alpha-Beta search and domain-specific move ordering.

> 📄 **Note:** All theoretical derivations, Squeeze-and-Excitation (SE-ResNet) architectures, training curves, checkmate game analysis, and detailed benchmark metrics are thoroughly presented in the accompanying IEEE-format academic paper: [`nom_bot_chess_paper.tex`](nom_bot_chess_paper.tex) (or compiled PDF).

---

## 🎮 1. For Players (Folder `play/`)

### A. Play Directly in the Terminal (CLI)
```bash
# 1. Install lightweight dependencies
pip install -r play/requirements.txt

# 2. Challenge the Bot in your command line!
python play/play.py
```

### B. Connect the Bot Online to Lichess (24/7 Matchmaking)
```bash
# Configure your Lichess API personal token and launch the bot
python play/lichess_bot_player.py
```
*(The engine also supports standard Universal Chess Interface (UCI) via `play/uci_engine.py`, allowing you to load it into graphical chess software such as Arena, Cutechess, or BanksiaGUI).*

---

## 🔬 2. For Training & Reproduction (Folder `training/`)

- **1-Click Google Colab Training:** Open [`training/chess_bot_elo1500_colab.ipynb`](training/chess_bot_elo1500_colab.ipynb) to train or benchmark directly on a free cloud GPU (T4).
- **Pre-Executed Results:** Inspect [`training/chess_bot_elo1500_colab_executed.ipynb`](training/chess_bot_elo1500_colab_executed.ipynb) to view full cell outputs, logs, and loss curves without re-running.
- **Standalone Local / Server CLI Training:**
  ```bash
  python training/train.py --max-games 25000 --epochs 8 --batch-size 256
  ```
- **Pre-trained Checkpoint:** [`training/best_chess_bot.pt`](training/best_chess_bot.pt) (~41.3 MB, Epoch 3 optimal checkpoint).

---

## 📂 Repository Structure

```text
c:\Nom_documents\chess_bot\
├── nom_bot_chess_paper.tex         # 📄 Full IEEE Academic Paper
├── README.md                      
├── .gitignore                      
│
├── play/                           # 🎮 Everything required to play against the bot
│   ├── play.py                     # Interactive terminal game client
│   ├── uci_engine.py               # UCI-compatible chess engine & ResNet model
│   ├── lichess_bot_player.py       # Online Lichess bot matchmaking client
│   └── requirements.txt            # Lightweight dependencies for playing
│
└── training/                       # 🔬 Everything required to train and reproduce
    ├── best_chess_bot.pt           # Pre-trained model weights (~41.3 MB)
    ├── chess_bot_elo1500_colab.ipynb # 1-click cloud training Colab notebook
    ├── chess_bot_elo1500_colab_executed.ipynb # Notebook with preserved cell outputs
    ├── train.py                    # Standalone CLI training script
    └── evaluation_report.txt       # Detailed training logs and test benchmarks
```
