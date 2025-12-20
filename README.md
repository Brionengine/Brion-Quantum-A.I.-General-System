# Brion Quantum AI Lab Website

A modern, high-performance website for Brion Quantum AI Lab (Brion.qt) featuring quantum-themed design, Rust/C backend, and advanced optimizations.

## Features

### Frontend
- **Modern Quantum Design**: Beautiful quantum-themed UI with gradient effects and animations
- **Responsive Layout**: Fully responsive design that works on all devices
- **Interactive Elements**: 
  - Animated quantum particle canvas
  - Smooth scroll animations
  - Interactive navigation
  - Dynamic statistics counters
- **API Integration**: Dynamic content loading from Rust backend

### Backend (Rust + C)
- **High-Performance Rust Server**: Actix-web with parallel processing
- **C Library Integration**: FFI bindings for performance-critical operations
- **Advanced Caching**: Redis with in-memory fallback
- **Parallel Processing**: Rayon for data parallelism
- **Optimized APIs**: Fast, cached endpoints with compression
- **CI/CD Pipeline**: Automated testing and deployment

### Performance Optimizations
- **Parallelism**: Multi-threaded processing with Rayon and C OpenMP
- **Caching**: Redis + in-memory cache with TTL management
- **Data Optimization**: Compression, string optimization, quantum computations
- **API Optimization**: Response caching, compression, connection pooling
- **Build Optimizations**: LTO, native CPU optimizations, profile-guided builds

## File Structure

```
Brion.qt website/
├── index.html              # Main HTML file
├── styles/
│   └── main.css            # Main stylesheet with quantum theme
├── scripts/
│   └── main.js             # JavaScript with API integration
├── backend/                 # Rust backend server
│   ├── src/
│   │   ├── main.rs         # Server entry point
│   │   ├── api.rs          # API endpoints
│   │   ├── cache.rs        # Cache management
│   │   ├── data.rs         # Data management
│   │   ├── ffi.rs          # C library bindings
│   │   └── utils.rs         # Utilities
│   ├── Cargo.toml          # Rust dependencies
│   ├── build.rs            # Build script
│   └── config.toml         # Configuration
├── c-lib/                  # C performance library
│   ├── src/
│   │   ├── quantum_compute.c
│   │   └── quantum_compute.h
│   └── Makefile
├── .github/workflows/
│   └── ci-cd.yml           # CI/CD pipeline
├── docker-compose.yml      # Docker setup
├── Dockerfile.backend      # Backend Docker image
├── nginx.conf              # Nginx configuration
├── SETUP.md                # Complete setup guide
└── README.md               # This file
```

## Setup & Deployment

### Quick Start (Docker - Recommended)

```bash
# Start all services (backend, Redis, Nginx)
docker-compose up -d

# Website available at http://localhost
# API available at http://localhost/api/v1
```

### Manual Setup

See [SETUP.md](SETUP.md) for complete instructions.

**Quick version:**
1. Build C library: `cd c-lib && make`
2. Start Redis: `docker run -d -p 6379:6379 redis:7-alpine`
3. Run backend: `cd backend && cargo run --release`
4. Serve frontend: `python3 -m http.server 8000`

### Domain Configuration (Brion.qt)

To deploy to your custom domain:

1. **Upload files** to your web hosting service
2. **Configure DNS** for Brion.qt to point to your hosting
3. **SSL Certificate**: Ensure HTTPS is enabled for security
4. **Update paths** if needed based on your hosting structure

### Recommended Hosting Options

- **Netlify**: Drag and drop deployment, automatic HTTPS
- **Vercel**: Fast deployment with custom domain support
- **GitHub Pages**: Free hosting for public repositories
- **Cloudflare Pages**: Fast CDN with custom domain support

## Customization

### Colors

Edit the CSS variables in `styles/main.css`:

```css
:root {
    --quantum-primary: #00d4ff;
    --quantum-secondary: #7c3aed;
    --quantum-accent: #06b6d4;
    /* ... */
}
```

### Content

Update content in `index.html`:
- Hero section text
- Research cards
- Project descriptions
- Contact information
- Social media links

### Animations

Adjust animation speeds and effects in:
- `styles/main.css` - CSS animations
- `scripts/main.js` - JavaScript animations

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Performance

- Optimized animations using CSS transforms
- Efficient canvas rendering
- Lazy loading for scroll animations
- Minimal dependencies (vanilla JavaScript)

## Future Enhancements

Potential additions:
- Blog section
- Research publications page
- Interactive quantum simulator
- Project showcase with detailed pages
- Contact form backend integration
- Multi-language support

## License

© 2024 Brion Quantum AI Lab. All rights reserved.

## Contact

- Website: Brion.qt
- GitHub: [@Brionengine](https://github.com/Brionengine)
- X (Twitter): [@Brionengine](https://x.com/Brionengine)

