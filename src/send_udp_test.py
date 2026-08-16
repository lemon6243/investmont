import json, math, socket, time

ADDR = ("127.0.0.1", 7755)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("UDP test -> 127.0.0.1:7755   Ctrl+C 로 종료")
t0 = time.time()
seq = 0

while True:
    t = time.time() - t0
    pts = []
    for i in range(50):
        a = i / 50.0 * math.tau
        pts += [
            40.0 * math.sin(a + t),          # X
            8.0 * (i - 25),                  # Y
            90.0 + 25.0 * math.cos(a + t),   # Z
        ]

    q = {
        "upperarm_l": [0, 0.2 * math.sin(t), 0, 0.98],
        "lowerarm_l": [0, 0, 0, 1],
        "hand_l":     [0, 0, 0, 1],
        "upperarm_r": [0, -0.2 * math.sin(t), 0, 0.98],
        "lowerarm_r": [0, 0, 0, 1],
        "hand_r":     [0, 0, 0, 1],
    }

    msg = json.dumps({"type": "frame", "seq": seq, "pts": pts, "q": q})
    sock.sendto(msg.encode("utf-8"), ADDR)
    seq += 1
    time.sleep(1 / 30)

