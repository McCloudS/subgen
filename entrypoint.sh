#!/bin/bash

# Default to 99:100 (nobody:users) if env vars not provided
PUID=${PUID:-99}
PGID=${PGID:-100}

echo "Starting with PUID: $PUID and PGID: $PGID"

# Create group if it doesn't exist
if ! getent group "$PGID" >/dev/null; then
    groupadd -g "$PGID" subgen
fi

# Create user if it doesn't exist
if ! getent passwd "$PUID" >/dev/null; then
    useradd -u "$PUID" -g "$PGID" -m -s /bin/bash subgen
fi

# Fix permissions on the application directory so the new user can write there
echo "Fixing permissions on /subgen..."
chown -R "$PUID":"$PGID" /subgen

# Drop privileges and execute the command provided in the Dockerfile (CMD)
echo "Starting application..."
exec gosu "$PUID":"$PGID" "$@"
