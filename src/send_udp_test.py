# send_udp_test.py
# UE 연결 테스트용. 데이터셋 필요 없음.
import json, math, socket, time

UE_IP, UE_PORT = "127.0.0.1", 7755
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"UDP 테스트 송신 -> {UE_IP}:{UE_PORT}")
print("UE Play 켠 뒤에 점이 원 그리며 움직이면 성공. 종료: Ctrl+C")

t = 0.0
while True:
    pts = []
    for i in range(50):
        x = 20.0 * math.sin(t + i * 0.12)
        y = i * 2.0 - 50.0
        z = 15.0 * math.cos(t * 0.7 + i * 0.05)
        pts.extend([x, y, z])

    msg = {
        "type": "frame",
        "word": "test",
        "i": int(t * 30) % 1000,
        "n": 1000,
        "pts": pts,
        "q": {
            "upperarm_l": [0.0, 0.2 * math.sin(t), 0.0, 0.98],
            "lowerarm_l": [0.0, 0.3 * math.sin(t + 0.4), 0.0, 0.95],
            "hand_l": [0.0, 0.0, 0.0, 1.0],
            "upperarm_r": [0.0, -0.2 * math.sin(t), 0.0, 0.98],
            "lowerarm_r": [0.0, -0.3 * math.sin(t + 0.4), 0.0, 0.95],
            "hand_r": [0.0, 0.0, 0.0, 1.0],
        },
    }
    sock.sendto(json.dumps(msg).encode("utf-8"), (UE_IP, UE_PORT))
    t += 1.0 / 30.0
    time.sleep(1.0 / 30.0)
