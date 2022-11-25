#!/bin/bash
set -e
USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}

service ssh restart
echo "KPConv with UID: $USER_ID, GID: $GROUP_ID"
usermod -u $USER_ID point_warrior
groupmod -g $GROUP_ID point_warrior
export HOME=/home/point_warrior
id point_warrior
exec /usr/sbin/gosu point_warrior /bin/bash
python3 tensorboard -logdir /home/point_warrior/tensorboard_logs
