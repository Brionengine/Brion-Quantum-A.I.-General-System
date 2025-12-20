# Quick Start Guide - Brion Quantum AI Lab

## 🚀 Fastest Way to Get Started

### Using Docker (Recommended - 2 minutes)

```bash
# Clone/navigate to project
cd "Brion.qt website"

# Start everything
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop everything
docker-compose down
```

**Access:**
- Website: http://localhost
- API: http://localhost/api/v1/health

### Manual Setup (5 minutes)

#### 1. Build C Library
```bash
cd c-lib
make -j$(nproc)
cd ..
```

#### 2. Start Redis (Optional)
```bash
docker run -d -p 6379:6379 --name brion-redis redis:7-alpine
```

#### 3. Start Backend
```bash
cd backend
cargo run --release
# Runs on http://localhost:8080
```

#### 4. Serve Frontend
```bash
# In project root
python3 -m http.server 8000
# Open http://localhost:8000
```

## 📋 Prerequisites

- **Docker** (for Docker setup) OR
- **Rust** 1.75+ (install from rustup.rs)
- **C Compiler** (gcc/clang)
- **Make**
- **Redis** (optional, for caching)

## 🔧 Configuration

### Backend Config
Edit `backend/config.toml`:
```toml
[server]
port = 8080

[cache]
redis_url = "redis://127.0.0.1:6379/"
```

### Frontend API URL
Edit `scripts/main.js`:
```javascript
const API_BASE_URL = 'http://localhost:8080/api/v1';
```

## 🧪 Test It

```bash
# Health check
curl http://localhost:8080/api/v1/health

# Get research data
curl http://localhost:8080/api/v1/research

# Get projects
curl http://localhost:8080/api/v1/projects
```

## 📚 Next Steps

- Read [SETUP.md](SETUP.md) for detailed setup
- Read [backend/README.md](backend/README.md) for backend details
- Read [c-lib/README.md](c-lib/README.md) for C library info

## 🐛 Troubleshooting

**C library not found?**
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/c-lib
```

**Redis connection failed?**
- Backend will use in-memory cache as fallback
- Check: `redis-cli ping`

**Port already in use?**
- Change port in `backend/config.toml`
- Or: `lsof -ti:8080 | xargs kill`

## 🎯 Production Deployment

1. Build: `docker-compose build`
2. Configure domain in `nginx.conf`
3. Set up SSL (Let's Encrypt)
4. Deploy: `docker-compose up -d`

For full production guide, see [SETUP.md](SETUP.md)

