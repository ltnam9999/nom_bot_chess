"""
Universal Chess Interface (UCI) wrapper for our Chess Bot.
This allows the bot to be loaded into any Chess GUI (Arena, Cutechess, BanksiaGUI)
and integrated with the standard 'lichess-bot' client (https://github.com/lichess-bot-devs/lichess-bot).
"""

import sys
import os
import torch
import torch.nn as nn
import chess
import numpy as np

# 1. Feature Encoding (same as notebook)
def encode_board(board: chess.Board) -> np.ndarray:
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    turn = board.turn
    piece_map = board.piece_map()
    for sq, piece in piece_map.items():
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        if not turn:
            rank = 7 - rank
        piece_idx = piece.piece_type - 1
        if piece.color == turn:
            planes[piece_idx, rank, file] = 1.0
        else:
            planes[6 + piece_idx, rank, file] = 1.0
            
    if turn:
        planes[12, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
        planes[13, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
        planes[14, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
        planes[15, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    else:
        planes[12, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
        planes[13, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
        planes[14, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
        planes[15, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
        
    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        if not turn:
            ep_rank = 7 - ep_rank
        planes[16, ep_rank, ep_file] = 1.0
        
    planes[17, :, :] = 1.0 if turn else 0.0
    return planes

def move_to_index(move: chess.Move, turn: bool) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    if not turn:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)
    return from_sq * 64 + to_sq

# 2. ResNet Architecture
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class ChessResNet(nn.Module):
    def __init__(self, in_channels=18, num_blocks=6, filters=128):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.ModuleList([ResBlock(filters) for _ in range(num_blocks)])
        self.policy_conv = nn.Sequential(
            nn.Conv2d(filters, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)
        self.value_conv = nn.Sequential(
            nn.Conv2d(filters, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.value_fc = nn.Sequential(
            nn.Linear(16 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.in_conv(x)
        for block in self.res_blocks:
            x = block(x)
        p = self.policy_conv(x).view(x.size(0), -1)
        policy_logits = self.policy_fc(p)
        v = self.value_conv(x).view(x.size(0), -1)
        value = self.value_fc(v)
        return policy_logits, value

# 3. Evaluation & Search
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

def load_model(weights_path=None):
    global model
    model = ChessResNet().to(device)
    if weights_path is None or not os.path.exists(weights_path):
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "training", "best_chess_bot.pt"),
            os.path.join(os.path.dirname(__file__), "best_chess_bot.pt"),
            os.path.join(os.path.dirname(__file__), "training", "best_chess_bot.pt"),
            "training/best_chess_bot.pt",
            "../training/best_chess_bot.pt",
            "best_chess_bot.pt"
        ]
        for c in candidates:
            if os.path.exists(c):
                weights_path = c
                break
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
    return model

def evaluate_board_hybrid(board: chess.Board) -> float:
    if board.is_checkmate():
        return -99999.0 if board.turn else 99999.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0
    mat_score = 0
    for sq, piece in board.piece_map().items():
        val = PIECE_VALUES[piece.piece_type]
        mat_score += val if piece.color == chess.WHITE else -val
    enc = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
    with torch.no_grad():
        _, val_out = model(enc)
        nn_val = val_out.item()
    white_perspective_nn = nn_val if board.turn == chess.WHITE else -nn_val
    return mat_score + white_perspective_nn * 300.0

def get_ordered_moves(board: chess.Board):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return []
    enc = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, _ = model(enc)
        policy_logits = policy_logits.squeeze(0)
    move_scores = []
    for m in legal_moves:
        idx = move_to_index(m, board.turn)
        score = policy_logits[idx].item()
        if board.is_capture(m):
            score += 2.0
        if board.gives_check(m):
            score += 1.0
        move_scores.append((score, m))
    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in move_scores]

def alphabeta(board: chess.Board, depth: int, alpha: float, beta: float, is_maximizing: bool) -> float:
    if depth == 0 or board.is_game_over():
        return evaluate_board_hybrid(board)
    ordered_moves = get_ordered_moves(board)
    if is_maximizing:
        max_eval = -float('inf')
        for move in ordered_moves:
            board.push(move)
            eval_score = alphabeta(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in ordered_moves:
            board.push(move)
            eval_score = alphabeta(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval

def select_best_move(board: chess.Board, search_depth: int = 2) -> chess.Move:
    ordered_moves = get_ordered_moves(board)
    if not ordered_moves:
        return None
    best_move = ordered_moves[0]
    is_white = (board.turn == chess.WHITE)
    alpha = -float('inf')
    beta = float('inf')
    if is_white:
        best_val = -float('inf')
        for move in ordered_moves:
            board.push(move)
            score = alphabeta(board, search_depth - 1, alpha, beta, False)
            board.pop()
            if score > best_val:
                best_val = score
                best_move = move
            alpha = max(alpha, score)
    else:
        best_val = float('inf')
        for move in ordered_moves:
            board.push(move)
            score = alphabeta(board, search_depth - 1, alpha, beta, True)
            board.pop()
            if score < best_val:
                best_val = score
                best_move = move
            beta = min(beta, score)
    return best_move

# 4. Main UCI Protocol Loop
def uci_main():
    board = chess.Board()
    load_model()
    
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        tokens = line.strip().split()
        if not tokens:
            continue
            
        cmd = tokens[0]
        if cmd == "uci":
            print("id name ResNetChessBot-1500")
            print("id author Antigravity & User")
            print("uciok")
            sys.stdout.flush()
        elif cmd == "isready":
            print("readyok")
            sys.stdout.flush()
        elif cmd == "ucinewgame":
            board.reset()
        elif cmd == "position":
            # Example: position startpos moves e2e4 e7e5 ...
            if len(tokens) > 1 and tokens[1] == "startpos":
                board.reset()
                if "moves" in tokens:
                    idx = tokens.index("moves")
                    for m_str in tokens[idx + 1:]:
                        board.push_uci(m_str)
            elif len(tokens) > 1 and tokens[1] == "fen":
                fen_parts = []
                idx = 2
                while idx < len(tokens) and tokens[idx] != "moves":
                    fen_parts.append(tokens[idx])
                    idx += 1
                board.set_fen(" ".join(fen_parts))
                if "moves" in tokens:
                    move_idx = tokens.index("moves")
                    for m_str in tokens[move_idx + 1:]:
                        board.push_uci(m_str)
        elif cmd == "go":
            best_m = select_best_move(board, search_depth=2)
            if best_m is None:
                # If no legal moves, pick anything
                legal = list(board.legal_moves)
                best_m = legal[0] if legal else chess.Move.null()
            print(f"bestmove {best_m.uci()}")
            sys.stdout.flush()
        elif cmd == "quit":
            break

if __name__ == "__main__":
    uci_main()
