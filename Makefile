.PHONY: help setup start start-gpu stop restart logs clean build build-cpu build-gpu test status

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Default target
help: ## Show this help message
	@echo "$(BLUE)SubGen - Docker Deployment Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  make setup          - Initial setup (copy .env.example to .env)"
	@echo "  make build          - Build CPU Docker image locally"
	@echo "  make build-gpu      - Build GPU Docker image locally"
	@echo ""
	@echo "$(GREEN)Operations:$(NC)"
	@echo "  make start          - Start SubGen (CPU)"
	@echo "  make start-gpu      - Start SubGen (GPU)"
	@echo "  make stop           - Stop SubGen"
	@echo "  make restart        - Restart SubGen"
	@echo "  make logs           - View logs (follow mode)"
	@echo "  make status         - Check container status"
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@echo "  make update         - Update to latest version"
	@echo "  make clean          - Stop and remove containers"
	@echo "  make test           - Test configuration"
	@echo ""
	@echo "$(YELLOW)Note: Edit .env file to configure SubGen before starting$(NC)"

setup: ## Initial setup - copy .env.example to .env
	@if [ -f .env ]; then \
		echo "$(YELLOW).env already exists. Skipping...$(NC)"; \
	else \
		cp .env.example .env; \
		echo "$(GREEN)Created .env file from .env.example$(NC)"; \
		echo "$(YELLOW)Please edit .env with your configuration before starting!$(NC)"; \
	fi

start: ## Start SubGen with CPU
	@echo "$(BLUE)Starting SubGen (CPU)...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(RED)Error: .env file not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi
	docker compose up -d
	@echo "$(GREEN)SubGen started!$(NC)"
	@echo "Visit http://localhost:9000/docs for API documentation"

start-gpu: ## Start SubGen with GPU support
	@echo "$(BLUE)Starting SubGen (GPU)...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(RED)Error: .env file not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi
	docker compose -f docker-compose.gpu.yml up -d
	@echo "$(GREEN)SubGen (GPU) started!$(NC)"
	@echo "Visit http://localhost:9000/docs for API documentation"

stop: ## Stop SubGen
	@echo "$(BLUE)Stopping SubGen...$(NC)"
	@docker compose down 2>/dev/null || docker compose -f docker-compose.gpu.yml down 2>/dev/null || true
	@echo "$(GREEN)SubGen stopped$(NC)"

restart: stop start ## Restart SubGen (CPU)

restart-gpu: stop start-gpu ## Restart SubGen (GPU)

logs: ## View SubGen logs (follow mode)
	@docker compose logs -f subgen 2>/dev/null || docker compose -f docker-compose.gpu.yml logs -f subgen 2>/dev/null || echo "$(RED)No running container found$(NC)"

status: ## Check SubGen status
	@echo "$(BLUE)Container Status:$(NC)"
	@docker ps | grep subgen || echo "$(YELLOW)No SubGen container running$(NC)"
	@echo ""
	@echo "$(BLUE)Testing HTTP endpoint:$(NC)"
	@curl -s http://localhost:9000/status 2>/dev/null | head -20 || echo "$(YELLOW)SubGen not responding on port 9000$(NC)"

build: ## Build CPU Docker image
	@echo "$(BLUE)Building SubGen CPU image...$(NC)"
	docker build -f Dockerfile.cpu -t subgen:cpu .
	@echo "$(GREEN)Build complete!$(NC)"

build-gpu: ## Build GPU Docker image
	@echo "$(BLUE)Building SubGen GPU image...$(NC)"
	docker build -f Dockerfile -t subgen:gpu .
	@echo "$(GREEN)Build complete!$(NC)"

clean: ## Stop and remove containers, networks, and volumes
	@echo "$(YELLOW)This will remove SubGen containers and volumes!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose down -v 2>/dev/null || docker compose -f docker-compose.gpu.yml down -v 2>/dev/null || true; \
		echo "$(GREEN)Cleaned up!$(NC)"; \
	else \
		echo "$(BLUE)Cancelled$(NC)"; \
	fi

update: ## Update SubGen to latest version
	@echo "$(BLUE)Updating SubGen...$(NC)"
	git pull
	docker compose pull 2>/dev/null || docker compose -f docker-compose.gpu.yml pull 2>/dev/null
	@echo "$(GREEN)Updated! Run 'make restart' to apply changes$(NC)"

test: ## Test configuration and dependencies
	@echo "$(BLUE)Testing configuration...$(NC)"
	@echo ""
	@echo "$(BLUE)1. Checking required files:$(NC)"
	@test -f .env && echo "  $(GREEN)✓$(NC) .env exists" || echo "  $(RED)✗$(NC) .env missing (run 'make setup')"
	@test -f Dockerfile && echo "  $(GREEN)✓$(NC) Dockerfile exists" || echo "  $(RED)✗$(NC) Dockerfile missing"
	@test -f Dockerfile.cpu && echo "  $(GREEN)✓$(NC) Dockerfile.cpu exists" || echo "  $(RED)✗$(NC) Dockerfile.cpu missing"
	@test -f docker-compose.yml && echo "  $(GREEN)✓$(NC) docker-compose.yml exists" || echo "  $(RED)✗$(NC) docker-compose.yml missing"
	@test -f docker-compose.gpu.yml && echo "  $(GREEN)✓$(NC) docker-compose.gpu.yml exists" || echo "  $(RED)✗$(NC) docker-compose.gpu.yml missing"
	@echo ""
	@echo "$(BLUE)2. Checking Docker:$(NC)"
	@command -v docker >/dev/null 2>&1 && echo "  $(GREEN)✓$(NC) Docker installed" || echo "  $(RED)✗$(NC) Docker not found"
	@docker compose version >/dev/null 2>&1 && echo "  $(GREEN)✓$(NC) Docker Compose v2 available" || echo "  $(RED)✗$(NC) Docker Compose not found"
	@echo ""
	@echo "$(BLUE)3. Checking GPU support (optional):$(NC)"
	@command -v nvidia-smi >/dev/null 2>&1 && echo "  $(GREEN)✓$(NC) NVIDIA drivers installed" || echo "  $(YELLOW)⚠$(NC) NVIDIA drivers not found (GPU mode unavailable)"
	@docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1 && echo "  $(GREEN)✓$(NC) Docker GPU support working" || echo "  $(YELLOW)⚠$(NC) Docker GPU support not available"
	@echo ""
	@echo "$(BLUE)4. Testing YAML syntax:$(NC)"
	@python3 -c "import yaml; yaml.safe_load(open('docker-compose.yml'))" && echo "  $(GREEN)✓$(NC) docker-compose.yml valid" || echo "  $(RED)✗$(NC) docker-compose.yml invalid"
	@python3 -c "import yaml; yaml.safe_load(open('docker-compose.gpu.yml'))" && echo "  $(GREEN)✓$(NC) docker-compose.gpu.yml valid" || echo "  $(RED)✗$(NC) docker-compose.gpu.yml invalid"
	@echo ""
	@if [ -f .env ]; then \
		echo "$(BLUE)5. Key configuration from .env:$(NC)"; \
		grep -E "^TRANSCRIBE_DEVICE=" .env | sed 's/^/  /' || echo "  $(YELLOW)⚠$(NC) TRANSCRIBE_DEVICE not set"; \
		grep -E "^WHISPER_MODEL=" .env | sed 's/^/  /' || echo "  $(YELLOW)⚠$(NC) WHISPER_MODEL not set"; \
		grep -E "^WEBHOOK_PORT=" .env | sed 's/^/  /' || echo "  $(YELLOW)⚠$(NC) WEBHOOK_PORT not set"; \
	fi
	@echo ""
	@echo "$(GREEN)Configuration test complete!$(NC)"

# Quick shortcuts
up: start
down: stop
ps: status
