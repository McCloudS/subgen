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

echo "Dropping root privileges and starting application..."
exec gosu "$PUID":"$PGID" "$@"
