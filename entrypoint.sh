#!/bin/bash

# Default to 99:100 (nobody:users) if env vars not provided
PUID=${PUID:-99}
PGID=${PGID:-100}

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

# ----------------------------------------------------------------
# PERMISSION FIXES
# This fixes the "Permission denied" errors for caches and app dirs
# ----------------------------------------------------------------
echo "Fixing permissions on /subgen..."
chown -R "$PUID":"$PGID" /subgen

echo "Fixing permissions on /cache..."
chown -R "$PUID":"$PGID" /cache

# If /tmp is mounted or used, ensure it is writable
chown -R "$PUID":"$PGID" /tmp

# ----------------------------------------------------------------
# START APP
# ----------------------------------------------------------------
echo "Dropping root privileges and starting application..."
exec gosu "$PUID":"$PGID" "$@"
