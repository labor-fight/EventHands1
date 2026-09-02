#!/usr/bin/env python3
"""SSH ProxyCommand helper: 通过 HTTP CONNECT 代理连到目标 host:port,再桥接 stdin/stdout。
用法(给 ssh 用): ssh -o ProxyCommand="python3 pxconnect.py %h %p" user@host
代理地址默认取环境变量 PXPROXY(形如 127.0.0.1:17911)。
"""
import os
import select
import socket
import sys


def main() -> int:
    if len(sys.argv) < 3:
        sys.stderr.write("usage: pxconnect.py <host> <port>\n")
        return 2
    host, port = sys.argv[1], sys.argv[2]
    proxy = os.environ.get("PXPROXY", "127.0.0.1:17911")
    ph, pp = proxy.split(":")

    s = socket.create_connection((ph, int(pp)), timeout=15)
    req = f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n"
    s.sendall(req.encode())

    # 读取代理响应头(直到空行),多余字节转给 stdout
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = s.recv(1)
        if not chunk:
            sys.stderr.write("pxconnect: proxy closed during CONNECT\n")
            return 1
        buf += chunk
    status_line = buf.split(b"\r\n", 1)[0].decode(errors="replace")
    if " 200 " not in status_line:
        sys.stderr.write(f"pxconnect: proxy refused: {status_line}\n")
        return 1
    leftover = buf.split(b"\r\n\r\n", 1)[1]
    if leftover:
        os.write(1, leftover)

    fin = sys.stdin.buffer.raw.fileno() if hasattr(sys.stdin.buffer, "raw") else sys.stdin.fileno()
    fout = 1
    s.setblocking(False)
    try:
        while True:
            r, _, _ = select.select([s, fin], [], [])
            if s in r:
                try:
                    data = s.recv(65536)
                except BlockingIOError:
                    data = b""
                if data == b"" and _sock_eof(s):
                    break
                if data:
                    os.write(fout, data)
            if fin in r:
                data = os.read(fin, 65536)
                if not data:
                    break
                _sendall(s, data)
    finally:
        s.close()
    return 0


def _sock_eof(s: socket.socket) -> bool:
    try:
        return s.recv(1, socket.MSG_PEEK) == b""
    except BlockingIOError:
        return False
    except OSError:
        return True


def _sendall(s: socket.socket, data: bytes) -> None:
    while data:
        try:
            n = s.send(data)
            data = data[n:]
        except BlockingIOError:
            select.select([], [s], [])


if __name__ == "__main__":
    sys.exit(main())
