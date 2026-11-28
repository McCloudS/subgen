#!/bin/bash

# ----------------------------------------------------------------
# SAFETY CHECK: PODMAN / ALREADY NON-ROOT
# ----------------------------------------------------------------
# If the container is already running as a non-root user (common in Podman --userns=keep-id
# or OpenShift), we cannot run groupadd/useradd. We just launch the app.
if [ "$(id -u)" -ne 0 ]; then
    echo "Running as non-root user ($(id -u)). Skipping permission setup."
    exec "$@"
fi

# Default to 99:100 (nobody:users) if env vars not provided
PUID=${PUID:-99}
PGID=${PGID:-100}

# ----------------------------------------------------------------
# CHECK: ROOTLESS PODMAN (Internal Root mapping)
# ----------------------------------------------------------------
# If PUID is set to 0, the user wants to run as root (which maps to Host User in Podman).
# We skip the user creation logic to avoid errors.
if [ "$PUID" -eq 0 ]; then
    echo "PUID set to 0. Running as internal root (Podman default behavior)."
    exec "$@"
fi

echo "-------------------------------------------------------"
echo "Initializing Container with PUID: $PUID and PGID: $PGID"
echo "-------------------------------------------------------"

# Create group if it doesn't exist
if ! getent group "$PGID" >/dev/null; then
    groupadd -g "$PGID" subgen
fi

# Create user if it doesn't exist
if ! getent passwd "$PUID" >/dev/null; then
    useradd -u "$PUID" -g "$PGID" -m -s /bin/bash subgen
fi

# Permission fixes
echo "Fixing permissions on /subgen..."
chown -R "$PUID":"$PGID" /subgen
echo "Fixing permissions on /cache..."
chown -R "$PUID":"$PGID" /cache
chown -R "$PUID":"$PGID" /tmp 2>/dev/null || true

# GPU device access for the unprivileged user. gosu resets supplementary groups (so docker
# group_add is lost), so add the user to the group of every GPU device node it can't otherwise
# open: /dev/dri (Intel/AMD Vulkan) and /dev/nvidia* (some distros, e.g. Gentoo, create these
# 0660 root:video instead of 0666). Without access whisper.cpp silently falls back to the CPU.
USERNAME=`getent passwd "$PUID" | cut -d: -f1`
for DEV in /dev/dri/renderD* /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm; do
  [ -e "$DEV" ] || continue
  MODE=`stat -c "%a" "$DEV"`
  case "${MODE: -1}" in 6|7) continue ;; esac   # already read/write for everyone
  GID=`stat -c "%g" "$DEV"`
  if [ "$GID" = "0" ]; then
    echo "Warning: $DEV is only accessible to root (mode $MODE). If the GPU isn't used, set PUID=0 and PGID=0."
    continue
  fi
  getent group "$GID" >/dev/null || groupadd -g "$GID" "gpu$GID"
  GROUP=`getent group "$GID" | cut -d: -f1`
  if ! id -nG "$USERNAME" | tr ' ' '\n' | grep -qx "$GROUP"; then
    echo "Adding $USERNAME to group $GROUP ($GID) for $DEV"
    usermod -aG "$GROUP" "$USERNAME"
  fi
done

echo "Dropping root privileges and starting application..."
exec gosu "$PUID" "$@"
