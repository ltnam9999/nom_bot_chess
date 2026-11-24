"""
Interactive CLI Chess Player against the trained ResNet Bot.
Run:
    python play.py
"""

import sys
import os
import time
import chess
from uci_engine import load_model, select_best_move

def print_board_pretty(board: chess.Board):
    UNICODE_PIECES = {
        'R': ' ♖ ', 'N': ' ♘ ', 'B': ' ♗ ', 'Q': ' ♕ ', 'K': ' ♔ ', 'P': ' ♙ ',
        'r': ' ♜ ', 'n': ' ♞ ', 'b': ' ♝ ', 'q': ' ♛ ', 'k': ' ♚ ', 'p': ' ♟ ',
        '.': ' · '
    }
    print("\n      a  b  c  d  e  f  g  h")
    print("    +------------------------+")
    for rank in range(7, -1, -1):
        row_str = f" {rank + 1}  |"
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            sym = piece.symbol() if piece else '.'
            row_str += UNICODE_PIECES.get(sym, ' . ')
        row_str += f"|  {rank + 1}"
        print(row_str)
    print("    +------------------------+")
    print("      a  b  c  d  e  f  g  h\n")

def main():
    print("=" * 60)
    print("♟️  CHESS BOT ELO 1500+ (Deep ResNet + Minimax Alpha-Beta)")
    print("=" * 60)
    print("Đang nạp mô hình mạng nơ-ron 'best_chess_bot.pt'...")
    
    weights_candidates = [
        os.path.join(os.path.dirname(__file__), "..", "training", "best_chess_bot.pt"),
        os.path.join(os.path.dirname(__file__), "training", "best_chess_bot.pt"),
        os.path.join(os.path.dirname(__file__), "best_chess_bot.pt"),
        "training/best_chess_bot.pt",
        "../training/best_chess_bot.pt",
        "best_chess_bot.pt"
    ]
    weights_path = None
    for c in weights_candidates:
        if os.path.exists(c):
            weights_path = c
            break

    if weights_path is None:
        print("❌ Lỗi: Không tìm thấy file trọng số 'best_chess_bot.pt' (trong thư mục gốc hoặc training/)!")
        return
        
    load_model(weights_path)
    print("✅ Nạp mô hình thành công!\n")

    color_choice = input("Bạn muốn cầm quân nào? (Nhập 'w' cho Trắng, 'b' cho Đen, mặc định: w): ").strip().lower()
    user_is_white = (color_choice != 'b')
    print(f"👉 Bạn cầm quân: {'TRẮNG' if user_is_white else 'ĐEN'}")
    print("👉 Hướng dẫn nhập nước đi: gõ dạng SAN (e4, Nf3, O-O, d5) hoặc UCI (e2e4, g1f3).")
    print("👉 Gõ 'quit' hoặc 'exit' để dừng chơi.\n")

    board = chess.Board()

    while not board.is_game_over():
        print_board_pretty(board)
        is_user_turn = (board.turn == chess.WHITE and user_is_white) or (board.turn == chess.BLACK and not user_is_white)

        if is_user_turn:
            while True:
                user_input = input(f"Lượt của BẠN ({'Trắng' if board.turn == chess.WHITE else 'Đen'}) > ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    print("Bạn đã thoát ván cờ.")
                    return
                try:
                    try:
                        move = board.parse_san(user_input)
                    except:
                        move = board.parse_uci(user_input)
                        
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("⚠️ Nước đi không hợp lệ! Hãy thử lại.")
                except Exception as e:
                    print(f"⚠️ Định dạng không đúng ({e}). Ví dụ hợp lệ: e4, Nf3, e2e4...")
        else:
            print("🤖 Bot đang suy nghĩ tính toán nước cờ tối ưu...")
            t0 = time.time()
            bot_move = select_best_move(board, search_depth=2)
            calc_time = time.time() - t0
            san_str = board.san(bot_move)
            board.push(bot_move)
            print(f"👉 🤖 Bot đã đi: {san_str} ({bot_move.uci()}) - [Thời gian: {calc_time:.2f}s]")

    print_board_pretty(board)
    print("=" * 60)
    print("🏁 VÁN ĐẤU KẾT THÚC!")
    print(f"Kết quả: {board.result()}")
    if board.outcome():
        print(f"Nguyên nhân: {board.outcome().termination.name}")
    print("=" * 60)

if __name__ == "__main__":
    main()
