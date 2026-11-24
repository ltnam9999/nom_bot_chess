"""
Lichess Bot Direct Runner
=========================
Kết nối trực tiếp mô hình Chess Bot của bạn với Lichess qua Lichess Bot API!
Không cần cài đặt phức tạp, chỉ cần Token Lichess Bot là bot có thể online 24/7.

Cách sử dụng:
1. Tạo 1 tài khoản Lichess mới (chưa từng chơi ván nào).
2. Vào https://lichess.org/account/oauth/token -> Tạo Personal API Access Token
   với các quyền:
   - "Play games with the bot API" (bot:play)
   - "Read incoming challenges" (challenge:read)
   - "Create, accept, decline challenges" (challenge:write)
3. Điền Token vào biến LICHESS_API_TOKEN bên dưới.
4. Chạy script: python lichess_bot_player.py
"""

import os
import sys
import json
import time
import threading
import requests
import chess
import torch
from uci_engine import load_model, select_best_move

# ========================================================
# CẤU HÌNH TOKEN VÀ ENGINE
# ========================================================
LICHESS_API_TOKEN = os.environ.get("LICHESS_BOT_TOKEN", "YOUR_LICHESS_TOKEN_HERE")
BASE_URL = "https://lichess.org"
HEADERS = {
    "Authorization": f"Bearer {LICHESS_API_TOKEN}"
}
BOT_SEARCH_DEPTH = 2  # Độ sâu tìm kiếm (2 hoặc 3, thời gian phản hồi ~0.2s)

def upgrade_account_to_bot():
    """Tự động nâng cấp tài khoản Lichess thường thành tài khoản BOT (chỉ làm 1 lần lúc đầu)"""
    url = f"{BASE_URL}/api/bot/account/upgrade"
    resp = requests.post(url, headers=HEADERS)
    if resp.status_code == 200:
        print("🎉 Nâng cấp tài khoản thành công lên BOT!")
    elif resp.status_code == 400 and "already" in resp.text.lower():
        print("Tài khoản này đã là tài khoản BOT.")
    else:
        print(f"Trạng thái nâng cấp: {resp.status_code} - {resp.text}")

def handle_game(game_id: str, bot_username: str):
    """Xử lý từng ván đấu trực tiếp theo thời gian thực (Real-time Stream)"""
    print(f"\n▶️ Bắt đầu ván đấu mới: {game_id}")
    url = f"{BASE_URL}/api/bot/game/stream/{game_id}"
    
    # Gửi lời chào trong khung chat Lichess
    try:
        requests.post(
            f"{BASE_URL}/api/bot/game/{game_id}/chat",
            headers=HEADERS,
            data={"room": "player", "text": "Hello! I am an AI Chess Bot powered by Deep ResNet. Good luck!"}
        )
    except:
        pass

    board = chess.Board()
    bot_is_white = False

    with requests.get(url, headers=HEADERS, stream=True) as stream:
        for line in stream.iter_lines():
            if not line:
                continue
            event = json.loads(line.decode('utf-8'))
            event_type = event.get("type")

            if event_type == "gameFull":
                white_player = event.get("white", {}).get("name", "")
                black_player = event.get("black", {}).get("name", "")
                bot_is_white = (white_player.lower() == bot_username.lower())
                print(f"Trận đấu: Trắng [{white_player}] vs Đen [{black_player}]")
                print(f"Bot cầm quân: {'TRẮNG' if bot_is_white else 'ĐEN'}")

                # Cập nhật các nước đi nếu vào giữa ván
                moves_str = event.get("state", {}).get("moves", "")
                board = chess.Board()
                if moves_str:
                    for m in moves_str.split():
                        board.push_uci(m)

            elif event_type == "gameState":
                moves_str = event.get("moves", "")
                board = chess.Board()
                if moves_str:
                    for m in moves_str.split():
                        board.push_uci(m)

                # Kiểm tra ván cờ kết thúc
                if event.get("status") not in ["started", "created"]:
                    print(f"🏁 Ván đấu {game_id} đã kết thúc. Trạng thái: {event.get('status')}")
                    break

            # Kiểm tra xem có phải lượt của Bot hay không
            is_bot_turn = (board.turn == chess.WHITE and bot_is_white) or (board.turn == chess.BLACK and not bot_is_white)
            
            if is_bot_turn and not board.is_game_over():
                print(f"🤖 Bot đang tính nước đi thứ {board.fullmove_number}...")
                t0 = time.time()
                best_move = select_best_move(board, search_depth=BOT_SEARCH_DEPTH)
                calc_time = time.time() - t0
                
                if best_move is not None:
                    move_uci = best_move.uci()
                    print(f"  👉 Bot chọn nước: {move_uci} ({calc_time:.2f}s)")
                    move_url = f"{BASE_URL}/api/bot/game/{game_id}/move/{move_uci}"
                    m_resp = requests.post(move_url, headers=HEADERS)
                    if m_resp.status_code != 200:
                        print(f"  ⚠️ Lỗi gửi nước đi: {m_resp.status_code} - {m_resp.text}")

def listen_events():
    """Lắng nghe các lời thách đấu và sự kiện ván đấu đến tài khoản Bot"""
    # Lấy thông tin tài khoản Bot
    user_resp = requests.get(f"{BASE_URL}/api/account", headers=HEADERS)
    if user_resp.status_code != 200:
        print("❌ LỖI XÁC THỰC: Token Lichess không hợp lệ hoặc đã hết hạn!")
        print(f"Chi tiết: {user_resp.status_code} - {user_resp.text}")
        return
        
    bot_info = user_resp.json()
    bot_username = bot_info.get("username")
    print("=" * 60)
    print(f"🤖 CHESS BOT ĐÃ ĐĂNG NHẬP THÀNH CÔNG VỚI TÊN: '{bot_username}'")
    print(f"🌐 Link hồ sơ trên Lichess: https://lichess.org/@/{bot_username}")
    print("📌 Bot đang ở chế độ lắng nghe thách đấu...")
    print("=" * 60)

    event_url = f"{BASE_URL}/api/stream/event"
    with requests.get(event_url, headers=HEADERS, stream=True) as stream:
        for line in stream.iter_lines():
            if not line:
                continue
            event = json.loads(line.decode('utf-8'))
            event_type = event.get("type")

            # 1. Có người thách đấu -> Tự động chấp nhận
            if event_type == "challenge":
                challenge = event.get("challenge", {})
                ch_id = challenge.get("id")
                challenger = challenge.get("challenger", {}).get("name", "Unknown")
                speed = challenge.get("speed", "standard")
                print(f"📩 Nhận lời thách đấu từ '{challenger}' (Thể thức: {speed})")

                # Chấp nhận lời thách đấu
                accept_url = f"{BASE_URL}/api/challenge/{ch_id}/accept"
                acc_resp = requests.post(accept_url, headers=HEADERS)
                if acc_resp.status_code == 200:
                    print(f"  ✅ Đã chấp nhận thách đấu từ {challenger}!")
                else:
                    print(f"  ❌ Không thể chấp nhận thách đấu: {acc_resp.text}")

            # 2. Ván đấu bắt đầu -> Tạo luồng riêng để đánh cờ
            elif event_type == "gameStart":
                game_info = event.get("game", {})
                game_id = game_info.get("id")
                game_thread = threading.Thread(target=handle_game, args=(game_id, bot_username))
                game_thread.daemon = True
                game_thread.start()

if __name__ == "__main__":
    if LICHESS_API_TOKEN == "YOUR_LICHESS_TOKEN_HERE":
        print("⚠️ HÃY ĐIỀN TOKEN LICHESS BOT CỦA BẠN!")
        print("Xem hướng dẫn trong file để lấy token từ Lichess.")
        sys.exit(1)
        
    print("Đang nạp mô hình mạng nơ-ron ChessResNet...")
    load_model("best_chess_bot.pt")
    print("Nạp mô hình thành công! Bắt đầu kết nối Lichess...")
    upgrade_account_to_bot()
    listen_events()
