"""
CDN Configuration for Static Asset Management
Supports CloudFlare and AWS CloudFront configurations
"""
import os
import hashlib
import mimetypes
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import aiofiles
import aiohttp
from pydantic import BaseModel, Field
from .config import settings

class CDNProvider(str, Enum):
    CLOUDFLARE = "cloudflare"
    CLOUDFRONT = "cloudfront"
    GENERIC = "generic"

@dataclass
class AssetManifest:
    """Asset manifest for tracking versioned assets"""
    files: Dict[str, str]  # original_path -> versioned_path
    hashes: Dict[str, str]  # file_path -> content_hash
    generated_at: datetime
    version: str

class CDNConfig(BaseModel):
    """CDN Configuration Model"""
    provider: CDNProvider = Field(default=CDNProvider.CLOUDFLARE)
    base_url: str = Field(default="https://cdn.example.com")
    zone_id: Optional[str] = Field(default=None)
    api_token: Optional[str] = Field(default=None)
    
    # Cache settings
    browser_cache_ttl: int = Field(default=31536000)  # 1 year for assets
    edge_cache_ttl: int = Field(default=2592000)      # 30 days at edge
    
    # Asset settings
    enable_versioning: bool = Field(default=True)
    enable_gzip: bool = Field(default=True)
    enable_brotli: bool = Field(default=True)
    
    # Origins
    origin_url: str = Field(default="https://api.example.com")
    fallback_origins: List[str] = Field(default_factory=list)
    
    # Security
    enable_security_headers: bool = Field(default=True)
    enable_cors: bool = Field(default=True)
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])
    
    class Config:
        env_prefix = "CDN_"

class StaticAssetManager:
    """Manages static assets for CDN deployment"""
    
    def __init__(self, config: CDNConfig):
        self.config = config
        self.manifest: Optional[AssetManifest] = None
        self.upload_session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.upload_session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.upload_session:
            await self.upload_session.close()
    
    def generate_asset_hash(self, content: bytes) -> str:
        """Generate content hash for asset versioning"""
        return hashlib.sha256(content).hexdigest()[:12]
    
    def get_versioned_filename(self, filepath: str, content_hash: str) -> str:
        """Generate versioned filename"""
        if not self.config.enable_versioning:
            return filepath
        
        path = Path(filepath)
        return f"{path.stem}.{content_hash}{path.suffix}"
    
    async def process_assets(self, asset_dir: Path) -> AssetManifest:
        """Process assets for CDN deployment"""
        files = {}
        hashes = {}
        
        # Common asset patterns
        asset_patterns = [
            "**/*.js", "**/*.css", "**/*.svg", "**/*.png", 
            "**/*.jpg", "**/*.jpeg", "**/*.gif", "**/*.webp",
            "**/*.woff", "**/*.woff2", "**/*.ttf", "**/*.otf",
            "**/*.mp4", "**/*.webm", "**/*.pdf"
        ]
        
        for pattern in asset_patterns:
            for file_path in asset_dir.glob(pattern):
                if file_path.is_file():
                    async with aiofiles.open(file_path, 'rb') as f:
                        content = await f.read()
                    
                    content_hash = self.generate_asset_hash(content)
                    relative_path = str(file_path.relative_to(asset_dir))
                    versioned_path = self.get_versioned_filename(relative_path, content_hash)
                    
                    files[relative_path] = versioned_path
                    hashes[relative_path] = content_hash
        
        manifest = AssetManifest(
            files=files,
            hashes=hashes,
            generated_at=datetime.utcnow(),
            version=f"v{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        )
        
        self.manifest = manifest
        return manifest
    
    async def upload_to_cloudflare(self, asset_dir: Path, manifest: AssetManifest):
        """Upload assets to CloudFlare R2 or KV"""
        if not self.config.api_token:
            raise ValueError("CloudFlare API token required")
        
        headers = {
            "Authorization": f"Bearer {self.config.api_token}",
            "Content-Type": "application/json"
        }
        
        for original_path, versioned_path in manifest.files.items():
            file_path = asset_dir / original_path
            
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            
            # Upload to CloudFlare R2 (example)
            upload_url = f"https://api.cloudflare.com/client/v4/accounts/{self.config.zone_id}/r2/buckets/assets/objects/{versioned_path}"
            
            content_type, _ = mimetypes.guess_type(str(file_path))
            upload_headers = {
                **headers,
                "Content-Type": content_type or "application/octet-stream"
            }
            
            if self.upload_session:
                async with self.upload_session.put(upload_url, data=content, headers=upload_headers) as response:
                    if response.status != 200:
                        print(f"Failed to upload {versioned_path}: {response.status}")
    
    async def upload_to_cloudfront(self, asset_dir: Path, manifest: AssetManifest):
        """Upload assets to AWS S3 for CloudFront"""
        # This would use boto3 async in real implementation
        # For now, provide structure
        
        for original_path, versioned_path in manifest.files.items():
            file_path = asset_dir / original_path
            
            # Configure S3 upload with proper headers
            cache_control = f"public, max-age={self.config.browser_cache_ttl}"
            content_type, _ = mimetypes.guess_type(str(file_path))
            
            # S3 upload would go here
            print(f"Would upload {original_path} as {versioned_path} to S3")
    
    async def purge_cache(self, paths: Optional[List[str]] = None):
        """Purge CDN cache"""
        if self.config.provider == CDNProvider.CLOUDFLARE:
            await self._purge_cloudflare_cache(paths)
        elif self.config.provider == CDNProvider.CLOUDFRONT:
            await self._purge_cloudfront_cache(paths)
    
    async def _purge_cloudflare_cache(self, paths: Optional[List[str]] = None):
        """Purge CloudFlare cache"""
        if not self.config.api_token or not self.config.zone_id:
            return
        
        url = f"https://api.cloudflare.com/client/v4/zones/{self.config.zone_id}/purge_cache"
        headers = {
            "Authorization": f"Bearer {self.config.api_token}",
            "Content-Type": "application/json"
        }
        
        payload = {"purge_everything": True} if not paths else {"files": paths}
        
        if self.upload_session:
            async with self.upload_session.post(url, json=payload, headers=headers) as response:
                return await response.json()
    
    async def _purge_cloudfront_cache(self, paths: Optional[List[str]] = None):
        """Purge CloudFront cache"""
        # Would use boto3 to create invalidation
        print(f"Would purge CloudFront cache for paths: {paths or 'all'}")

class CDNMiddleware:
    """FastAPI middleware for CDN integration"""
    
    def __init__(self, config: CDNConfig, manifest_path: Optional[Path] = None):
        self.config = config
        self.manifest_path = manifest_path
        self.manifest: Optional[AssetManifest] = None
        self.load_manifest()
    
    def load_manifest(self):
        """Load asset manifest"""
        if self.manifest_path and self.manifest_path.exists():
            import json
            with open(self.manifest_path) as f:
                data = json.load(f)
                self.manifest = AssetManifest(
                    files=data['files'],
                    hashes=data['hashes'],
                    generated_at=datetime.fromisoformat(data['generated_at']),
                    version=data['version']
                )
    
    def get_asset_url(self, path: str) -> str:
        """Get CDN URL for asset"""
        if self.manifest and path in self.manifest.files:
            versioned_path = self.manifest.files[path]
            return f"{self.config.base_url}/{versioned_path}"
        return f"{self.config.base_url}/{path}"
    
    def add_security_headers(self, response):
        """Add security headers for CDN responses"""
        if not self.config.enable_security_headers:
            return
        
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        if self.config.enable_cors:
            origin_header = "*" if "*" in self.config.allowed_origins else ",".join(self.config.allowed_origins)
            response.headers["Access-Control-Allow-Origin"] = origin_header

# Configuration instances
def get_cdn_config() -> CDNConfig:
    """Get CDN configuration from environment"""
    return CDNConfig(
        provider=os.getenv("CDN_PROVIDER", "cloudflare"),
        base_url=os.getenv("CDN_BASE_URL", "https://cdn.automata-app.com"),
        zone_id=os.getenv("CDN_ZONE_ID"),
        api_token=os.getenv("CDN_API_TOKEN"),
        origin_url=os.getenv("CDN_ORIGIN_URL", "https://api.automata-app.com"),
        browser_cache_ttl=int(os.getenv("CDN_BROWSER_CACHE_TTL", "31536000")),
        edge_cache_ttl=int(os.getenv("CDN_EDGE_CACHE_TTL", "2592000"))
    )

# Deployment script
async def deploy_assets():
    """Deploy assets to CDN"""
    config = get_cdn_config()
    asset_dir = Path(__file__).parent.parent.parent / "frontend" / "dist"
    
    if not asset_dir.exists():
        print("Frontend build directory not found. Run 'npm run build' first.")
        return
    
    async with StaticAssetManager(config) as manager:
        manifest = await manager.process_assets(asset_dir)
        
        # Save manifest
        manifest_path = asset_dir / "asset-manifest.json"
        manifest_data = {
            "files": manifest.files,
            "hashes": manifest.hashes,
            "generated_at": manifest.generated_at.isoformat(),
            "version": manifest.version
        }
        
        async with aiofiles.open(manifest_path, 'w') as f:
            import json
            await f.write(json.dumps(manifest_data, indent=2))
        
        # Upload based on provider
        if config.provider == CDNProvider.CLOUDFLARE:
            await manager.upload_to_cloudflare(asset_dir, manifest)
        elif config.provider == CDNProvider.CLOUDFRONT:
            await manager.upload_to_cloudfront(asset_dir, manifest)
        
        # Purge cache
        await manager.purge_cache()
        
        print(f"Deployed {len(manifest.files)} assets to CDN")
        print(f"Manifest version: {manifest.version}")

if __name__ == "__main__":
    asyncio.run(deploy_assets())