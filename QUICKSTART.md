# SubGen Quick Start Guide

Get SubGen running in 5 minutes or less!

## Prerequisites

- Docker & Docker Compose installed
- For GPU: NVIDIA GPU with drivers and nvidia-container-toolkit

## Option 1: Using Makefile (Easiest)

```bash
# 1. Clone repository
git clone https://github.com/McCloudS/subgen.git
cd subgen

# 2. Setup configuration
make setup

# 3. Edit your .env file with your settings
nano .env
# At minimum, set:
# - TRANSCRIBE_DEVICE (cpu or gpu)
# - TV_PATH and MOVIES_PATH
# - PLEX_TOKEN and PLEX_SERVER (or JELLYFIN_* equivalents)

# 4. Start SubGen
make start           # For CPU
# OR
make start-gpu       # For GPU

# 5. Check status
make status

# 6. View logs
make logs
```

**Available commands:**
```bash
make help      # Show all available commands
make start     # Start SubGen (CPU)
make start-gpu # Start SubGen (GPU)
make stop      # Stop SubGen
make restart   # Restart SubGen
make logs      # View logs
make status    # Check status
make test      # Test configuration
make update    # Update to latest version
```

## Option 2: Using Docker Compose

### CPU Version

```bash
# 1. Setup
cp .env.example .env
nano .env  # Configure your settings

# 2. Start
docker compose up -d

# 3. Check logs
docker compose logs -f
```

### GPU Version

```bash
# 1. Setup
cp .env.example .env
nano .env  # Configure your settings

# 2. Start with GPU
docker compose -f docker-compose.gpu.yml up -d

# 3. Check logs
docker compose logs -f
```

## Option 3: Using Docker Run

### CPU Version

```bash
docker run -d \
  --name subgen \
  --restart unless-stopped \
  -p 9000:9000 \
  -v /path/to/tv:/tv \
  -v /path/to/movies:/movies \
  -v ./models:/subgen/models \
  -e WHISPER_MODEL=medium \
  -e TRANSCRIBE_DEVICE=cpu \
  -e PLEX_SERVER=http://192.168.1.100:32400 \
  -e PLEX_TOKEN=your_token_here \
  mccloud/subgen:cpu
```

### GPU Version

```bash
docker run -d \
  --name subgen-gpu \
  --restart unless-stopped \
  --gpus all \
  -p 9000:9000 \
  -v /path/to/tv:/tv \
  -v /path/to/movies:/movies \
  -v ./models:/subgen/models \
  -e WHISPER_MODEL=medium \
  -e TRANSCRIBE_DEVICE=gpu \
  -e PLEX_SERVER=http://192.168.1.100:32400 \
  -e PLEX_TOKEN=your_token_here \
  mccloud/subgen:latest
```

## Essential Configuration

Edit your `.env` file with these required settings:

```env
# Device: cpu, gpu, or cuda
TRANSCRIBE_DEVICE=cpu

# Model size: tiny, small, medium, large-v3, distil-large-v3
WHISPER_MODEL=medium

# Media paths (must match your media server)
TV_PATH=/path/to/your/tv
MOVIES_PATH=/path/to/your/movies
MODELS_PATH=./models

# Plex (get token from https://support.plex.tv/articles/204059436)
PLEX_SERVER=http://192.168.1.100:32400
PLEX_TOKEN=your_plex_token_here

# OR Jellyfin
JELLYFIN_SERVER=http://192.168.1.100:8096
JELLYFIN_TOKEN=your_jellyfin_token_here
```

## Configure Webhooks

### Plex
1. Plex Settings → Webhooks → Add Webhook
2. URL: `http://YOUR_SERVER_IP:9000/plex`

### Jellyfin
1. Install Webhook plugin
2. Add Generic Destination: `http://YOUR_SERVER_IP:9000/jellyfin`
3. Enable: Item Added, Playback Start
4. Add Header: `Content-Type: application/json`

### Emby
1. Settings → Webhooks → Add
2. URL: `http://YOUR_SERVER_IP:9000/emby`
3. Content Type: `multipart/form-data`

### Bazarr
1. Settings → Whisper Provider
2. Endpoint: `http://YOUR_SERVER_IP:9000`

## Verify It's Working

```bash
# Check API status
curl http://localhost:9000/status

# View API documentation
# Open browser: http://localhost:9000/docs

# Check logs
docker logs -f subgen
# OR
make logs
```

## Troubleshooting

**Container won't start?**
```bash
docker logs subgen
make test  # Check configuration
```

**Can't reach SubGen from media server?**
- Use server IP, not `localhost` or `127.0.0.1`
- Ensure firewall allows port 9000
- Check `docker ps` to verify container is running

**Subtitles not generating?**
- Enable `DEBUG=True` in .env
- Check logs for errors
- Verify media paths match between SubGen and media server
- Test manually: http://localhost:9000/docs → POST /batch

**GPU not working?**
```bash
# Verify GPU is accessible
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

## Model Selection

| Model | VRAM | Speed | Quality | Recommended For |
|-------|------|-------|---------|----------------|
| tiny | 1GB | Fastest | Basic | Testing only |
| small | 2GB | Fast | Good | Quick jobs |
| medium | 5GB | Moderate | Very Good | **Most users** |
| large-v3 | 10GB | Slow | Excellent | Best quality |
| distil-large-v3 | 5GB | Fast | Excellent | **Best balance** |

**Recommendation:** Start with `medium` or `distil-large-v3`

## Next Steps

- Read [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for advanced configuration
- Read [README.md](README.md) for full documentation
- Visit http://localhost:9000/docs for API documentation

## Common Commands

```bash
# Using Makefile
make start         # Start
make stop          # Stop
make restart       # Restart
make logs          # View logs
make status        # Check status
make update        # Update to latest

# Using Docker Compose
docker compose up -d              # Start
docker compose down               # Stop
docker compose restart            # Restart
docker compose logs -f            # Logs
docker compose pull && docker compose up -d  # Update
```

## Support

- **Issues:** https://github.com/McCloudS/subgen/issues
- **Full Docs:** [README.md](README.md)
- **Deployment Guide:** [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)
