#!/bin/bash
# Start-up of the RailTwin-KPConv container:
#   1. adapt the uid / gid of point_warrior to the host user (files in the mounted folders keep their owner)
#   2. ssh with the key(s) from SSH_AUTHORIZED_KEYS (no password login)
#   3. compile the C++ wrappers in the mounted repository once
#   4. TensorBoard on port 6006 (logs below TENSORBOARD_LOGDIR)
#   5. run the command (default: bash) as point_warrior
set -e
USER_NAME=point_warrior
USER_HOME=/home/${USER_NAME}
REPO=${USER_HOME}/railtwin_kpconv
USER_ID=${LOCAL_UID:-1000}
GROUP_ID=${LOCAL_GID:-1000}

if [ "$(id -u ${USER_NAME})" != "${USER_ID}" ] || [ "$(id -g ${USER_NAME})" != "${GROUP_ID}" ]; then
    groupmod -o -g "${GROUP_ID}" ${USER_NAME}
    usermod -o -u "${USER_ID}" -g "${GROUP_ID}" ${USER_NAME}
    chown "${USER_ID}:${GROUP_ID}" ${USER_HOME}
fi
echo "RailTwin-KPConv as ${USER_NAME} (uid ${USER_ID}, gid ${GROUP_ID})"

if [ -n "${SSH_AUTHORIZED_KEYS}" ]; then
    install -d -m 700 -o "${USER_ID}" -g "${GROUP_ID}" ${USER_HOME}/.ssh
    printf '%s\n' "${SSH_AUTHORIZED_KEYS}" > ${USER_HOME}/.ssh/authorized_keys
    chown "${USER_ID}:${GROUP_ID}" ${USER_HOME}/.ssh/authorized_keys
    chmod 600 ${USER_HOME}/.ssh/authorized_keys
    service ssh start > /dev/null
    echo "ssh: port 22 (host port see docker-compose.yml), key login only"
else
    echo "ssh: disabled (set SSH_AUTHORIZED_KEYS to enable)"
fi

if [ -d "${REPO}/cpp_wrappers" ] && ! ls ${REPO}/cpp_wrappers/cpp_subsampling/grid_subsampling*.so > /dev/null 2>&1; then
    echo "Compiling the C++ wrappers ..."
    gosu ${USER_NAME} bash -c "cd ${REPO}/cpp_wrappers && bash compile_wrappers.sh" > /tmp/compile_wrappers.log 2>&1 || true
    if ls ${REPO}/cpp_wrappers/cpp_subsampling/grid_subsampling*.so ${REPO}/cpp_wrappers/cpp_neighbors/radius_neighbors*.so \
            > /dev/null 2>&1; then
        echo "C++ wrappers compiled"
    else
        echo "C++ wrappers FAILED, last lines of /tmp/compile_wrappers.log:"
        tail -5 /tmp/compile_wrappers.log
    fi
fi

TENSORBOARD_LOGDIR=${TENSORBOARD_LOGDIR:-${USER_HOME}/workdir}
gosu ${USER_NAME} tensorboard --logdir "${TENSORBOARD_LOGDIR}" --bind_all --port 6006 > /tmp/tensorboard.log 2>&1 &
echo "TensorBoard: port 6006 (host port see docker-compose.yml), logs from ${TENSORBOARD_LOGDIR}"

export HOME=${USER_HOME}
exec gosu ${USER_NAME} "$@"
