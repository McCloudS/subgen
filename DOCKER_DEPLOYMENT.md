# SubGen Docker Deployment Guide

This guide will help you deploy SubGen using Docker or Docker Compose for production use.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Deployment Options](#deployment-options)
  - [CPU-Only Deployment](#cpu-only-deployment)
  - [GPU/CUDA Deployment](#gpucuda-deployment)
  - [Using Pre-built Images](#using-pre-built-images)
- [Configuration](#configuration)
- [Media Server Setup](#media-server-setup)
- [Maintenance](#maintenance)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required
- Docker Engine 20.10+ or Docker Desktop
- Docker Compose v2.0+
- Sufficient disk space for Whisper models (1-10GB depending on model size)

### For GPU Support
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- CUDA Toolkit 12.3+ drivers

**Install NVIDIA Container Toolkit:**
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## Quick Start

### 1. Clone or Download Repository

```bash
git clone https://github.com/McCloudS/subgen.git
cd subgen
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your settings
nano .env  # or vim, code, etc.
```

**Minimum required configuration:**
```env
# Set device type
TRANSCRIBE_DEVICE=cpu  # or 'gpu' for NVIDIA

# Set your media paths
TV_PATH=/path/to/your/tv/shows
MOVIES_PATH=/path/to/your/movies
MODELS_PATH=/path/to/store/models

# Configure media server (example for Plex)
PLEX_SERVER=http://192.168.1.100:32400
PLEX_TOKEN=your_plex_token_here
```

### 3. Start SubGen

**For CPU:**
```bash
docker compose up -d
```

**For GPU:**
```bash
docker compose -f docker-compose.gpu.yml up -d
```

### 4. Verify It's Running

```bash
# Check logs
docker compose logs -f subgen

# Check status endpoint
curl http://localhost:9000/status

# View API docs
# Open browser to: http://localhost:9000/docs
```

---

## Deployment Options

### CPU-Only Deployment

Best for: Low-power servers, home labs, testing

**Using Docker Compose:**
```bash
docker compose up -d
```

**Using Docker Run:**
```bash
docker run -d \
  --name subgen \
  --restart unless-stopped \
  -p 9000:9000 \
  -v /path/to/tv:/tv \
  -v /path/to/movies:/movies \
  -v /path/to/models:/subgen/models \
  -e WHISPER_MODEL=medium \
  -e TRANSCRIBE_DEVICE=cpu \
  -e PLEX_SERVER=http://plex:32400 \
  -e PLEX_TOKEN=your_token \
  mccloud/subgen:cpu
```

**Build locally:**
```bash
docker build -f Dockerfile.cpu -t subgen:cpu .
```

### GPU/CUDA Deployment

Best for: Servers with NVIDIA GPUs, faster transcription

**Using Docker Compose:**
```bash
docker compose -f docker-compose.gpu.yml up -d
```

**Using Docker Run:**
```bash
docker run -d \
  --name subgen-gpu \
  --restart unless-stopped \
  --gpus all \
  -p 9000:9000 \
  -v /path/to/tv:/tv \
  -v /path/to/movies:/movies \
  -v /path/to/models:/subgen/models \
  -e WHISPER_MODEL=medium \
  -e TRANSCRIBE_DEVICE=gpu \
  -e PLEX_SERVER=http://plex:32400 \
  -e PLEX_TOKEN=your_token \
  mccloud/subgen:latest
```

**Build locally:**
```bash
docker build -f Dockerfile -t subgen:gpu .
```

**Verify GPU access:**
```bash
docker exec subgen-gpu nvidia-smi
```

### Using Pre-built Images

SubGen provides official images on Docker Hub:

| Image | Use Case | Size |
|-------|----------|------|
| `mccloud/subgen:latest` | GPU/CUDA support | ~8GB |
| `mccloud/subgen:cpu` | CPU-only, smaller | ~4GB |
| `mccloud/subgen:cuda` | Alias for latest | ~8GB |

**Pull and run:**
```bash
# CPU version
docker pull mccloud/subgen:cpu
docker run -d --name subgen mccloud/subgen:cpu

# GPU version
docker pull mccloud/subgen:latest
docker run -d --name subgen --gpus all mccloud/subgen:latest
```

---

## Configuration

### Essential Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSCRIBE_DEVICE` | `cpu` | Device to use: `cpu`, `gpu`, or `cuda` |
| `WHISPER_MODEL` | `medium` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3`, etc. |
| `WEBHOOK_PORT` | `9000` | Port for webhook server |
| `MODEL_PATH` | `/subgen/models` | Where to store downloaded models |

### Model Selection Guide

| Model | Size | VRAM | Speed | Accuracy | Best For |
|-------|------|------|-------|----------|----------|
| `tiny` | ~75MB | ~1GB | Fastest | Low | Testing |
| `base` | ~150MB | ~1GB | Very Fast | Fair | Quick jobs |
| `small` | ~500MB | ~2GB | Fast | Good | Most users |
| `medium` | ~1.5GB | ~5GB | Moderate | Very Good | Recommended |
| `large-v3` | ~3GB | ~10GB | Slow | Excellent | Best quality |
| `distil-large-v3` | ~1.5GB | ~5GB | Fast | Excellent | Best balance |

**Recommendation:** Start with `medium` or `distil-large-v3` for best balance of speed and quality.

### Path Mapping

SubGen **must** see files at the same path as your media server, OR use path mapping:

**Scenario 1: Same Paths (Recommended)**
```yaml
# Plex sees: /tv/Show/episode.mkv
# SubGen sees: /tv/Show/episode.mkv
volumes:
  - /mnt/media/tv:/tv
  - /mnt/media/movies:/movies
```

**Scenario 2: Different Paths (Use Mapping)**
```yaml
# Plex sees: /data/tv/Show/episode.mkv
# SubGen sees: /tv/Show/episode.mkv
environment:
  - USE_PATH_MAPPING=True
  - PATH_MAPPING_FROM=/data/tv
  - PATH_MAPPING_TO=/tv
volumes:
  - /mnt/media/tv:/tv
```

---

## Media Server Setup

### Plex

1. **Get Plex Token:**
   - Visit: https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/

2. **Configure Webhook:**
   - Plex Settings → Webhooks → Add Webhook
   - URL: `http://YOUR_SERVER_IP:9000/plex`

3. **Update .env:**
   ```env
   PLEX_SERVER=http://192.168.1.100:32400
   PLEX_TOKEN=your_token_here
   ```

### Jellyfin

1. **Install Webhook Plugin:**
   - Dashboard → Plugins → Catalog → Webhooks

2. **Configure Webhook:**
   - Add Generic Destination
   - URL: `http://YOUR_SERVER_IP:9000/jellyfin`
   - Events: Item Added, Playback Start
   - Enable "Send All Properties"
   - Add Header: `Content-Type: application/json`

3. **Get API Token:**
   - Dashboard → API Keys → Add New Key

4. **Update .env:**
   ```env
   JELLYFIN_SERVER=http://192.168.1.100:8096
   JELLYFIN_TOKEN=your_token_here
   ```

### Emby

1. **Configure Webhook:**
   - Settings → Webhooks → Add Webhook
   - URL: `http://YOUR_SERVER_IP:9000/emby`
   - Content Type: `multipart/form-data`
   - Events: New Media Added, Start, Unpause

**No token needed!** Emby provides full info in webhook.

### Bazarr (Whisper Provider)

1. **Configure Provider:**
   - Settings → Subtitles → Whisper Provider
   - Endpoint: `http://YOUR_SERVER_IP:9000`
   - Language: Select desired languages

2. **No additional SubGen config needed!**

---

## Maintenance

### Viewing Logs

```bash
# Follow logs in real-time
docker compose logs -f subgen

# View last 100 lines
docker compose logs --tail=100 subgen

# Save logs to file
docker compose logs subgen > subgen.log
```

### Updating SubGen

**Option 1: Auto-update (Easiest)**
```env
# In .env file
UPDATE=True
```
Restart container to pull latest code.

**Option 2: Manual Docker image update**
```bash
# Stop container
docker compose down

# Pull latest image
docker compose pull

# Start with new image
docker compose up -d
```

**Option 3: Rebuild from source**
```bash
# Pull latest code
git pull

# Rebuild and restart
docker compose up -d --build
```

### Backup Configuration

```bash
# Backup environment config
cp .env .env.backup

# Backup models (optional, they can be re-downloaded)
tar -czf models-backup.tar.gz models/
```

### Monitoring Health

```bash
# Check container health
docker ps

# Test status endpoint
curl http://localhost:9000/status

# Check API docs
curl http://localhost:9000/docs
```

---

## Troubleshooting

### Container Won't Start

**Check logs:**
```bash
docker compose logs subgen
```

**Common issues:**
- Missing environment variables → Check `.env` file exists
- Port conflict → Change `WEBHOOK_PORT` in `.env`
- Volume permissions → Ensure Docker can access media paths

### GPU Not Detected

```bash
# Verify NVIDIA drivers
nvidia-smi

# Check Docker can see GPU
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi

# Verify SubGen container has GPU access
docker exec subgen-gpu nvidia-smi
```

### Media Server Can't Reach SubGen

**Test connectivity:**
```bash
# From media server host
curl http://SUBGEN_IP:9000/status

# Check Docker network
docker network ls
docker network inspect subgen_default
```

**Solutions:**
- Ensure firewall allows port 9000
- Use host IP, not `localhost` (if media server is in different container)
- Check Docker network mode

### Subtitles Not Generated

**Enable debug logging:**
```env
DEBUG=True
```

**Check:**
1. Media paths match between SubGen and media server
2. File doesn't already have subtitles (check skip settings)
3. Webhook is correctly configured
4. Check logs for error messages

**Test manually:**
```bash
# Use batch endpoint to test a specific file
curl -X POST "http://localhost:9000/batch" \
  -H "Content-Type: application/json" \
  -d '{"path": "/tv/Show/episode.mkv"}'
```

### High CPU/Memory Usage

**Reduce resource usage:**
```env
# Reduce concurrent jobs
CONCURRENT_TRANSCRIPTIONS=1

# Use smaller model
WHISPER_MODEL=small

# Reduce threads
WHISPER_THREADS=2
```

### Models Re-downloading

**Ensure persistent volume:**
```yaml
volumes:
  - ./models:/subgen/models  # Must persist between restarts
```

---

## Production Best Practices

### 1. Use Docker Volumes for Models
```yaml
volumes:
  - subgen-models:/subgen/models

volumes:
  subgen-models:
    driver: local
```

### 2. Set Resource Limits
```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      memory: 4G
```

### 3. Enable Logging Driver
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### 4. Use Docker Secrets for Tokens
```bash
echo "your_token" | docker secret create plex_token -

# Reference in compose
secrets:
  - plex_token
environment:
  - PLEX_TOKEN_FILE=/run/secrets/plex_token
```

### 5. Run Behind Reverse Proxy
```nginx
# nginx example
location /subgen/ {
    proxy_pass http://localhost:9000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

---

## Advanced Usage

### Multiple Media Server Support

You can configure SubGen to work with multiple media servers simultaneously:

```env
# Enable multiple servers
PLEX_SERVER=http://plex:32400
PLEX_TOKEN=plex_token

JELLYFIN_SERVER=http://jellyfin:8096
JELLYFIN_TOKEN=jellyfin_token

# Both will send webhooks to same SubGen instance
```

### Batch Processing Existing Media

```env
# Set folders to scan
TRANSCRIBE_FOLDERS=/tv|/movies

# Enable monitoring for new files
MONITOR=True
```

Then trigger via API:
```bash
curl -X POST "http://localhost:9000/batch" \
  -H "Content-Type: application/json" \
  -d '{"path": "/tv"}'
```

---

## Support

- **Issues:** https://github.com/McCloudS/subgen/issues
- **Discussions:** https://github.com/McCloudS/subgen/discussions
- **Documentation:** https://github.com/McCloudS/subgen/blob/main/README.md

---

## License

SubGen is open source software. See [LICENSE](LICENSE) file for details.
