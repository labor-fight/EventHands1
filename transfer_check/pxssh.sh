#!/bin/sh
# 供 rsync/ssh 使用的包装:走 Cursor 本地 CONNECT 代理连到目标机。
exec ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=30 \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=8 \
    -o "ProxyCommand=python3 /data1/lyq/code/mesh/EventHands1/transfer_check/pxconnect.py %h %p" \
    "$@"
