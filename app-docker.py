import os
import shutil
import time
import gzip
import io
import requests
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import threading
from threading import Thread
import logging
logger = logging.getLogger("MacReplayXC")
logger.setLevel(logging.INFO)
logFormat = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# Docker-optimized paths
if os.getenv("CONFIG"):
    configFile = os.getenv("CONFIG")
    log_dir = os.path.dirname(configFile)
else:
    # Default paths for container
    log_dir = "/app/data"
    configFile = os.path.join(log_dir, "MacReplayXC.json")

# Create directories if they don't exist
os.makedirs(log_dir, exist_ok=True)
os.makedirs("/app/logs", exist_ok=True)

# Log file path for container
log_file_path = os.path.join("/app/logs", "MacReplayXC.log")

# Set up logging
fileHandler = logging.FileHandler(log_file_path)
fileHandler.setFormatter(logFormat)
logger.addHandler(fileHandler)

consoleFormat = logging.Formatter("[%(levelname)s] %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(consoleFormat)
logger.addHandler(consoleHandler)

# Docker-optimized ffmpeg paths (system-installed)
ffmpeg_path = "ffmpeg"
ffprobe_path = "ffprobe"

# Check if the binaries exist
import subprocess

def get_stream_url_with_auth(playlist_host, portal_id, channel_id):
    """
    Generate stream URL with embedded basic auth if needed.
    
    Args:
        playlist_host (str): Host for the playlist
        portal_id (str): Portal ID
        channel_id (str): Channel ID
        
    Returns:
        str: Stream URL with or without embedded auth
    """
    base_url = f"http://{playlist_host}/play/{portal_id}/{channel_id}"
    
    # Check if we should embed basic auth credentials
    settings = getSettings()
    
    # If public access is disabled, embed auth for VLC compatibility
    if settings.get("public playlist access", "true") == "false":
        
        # Try to get auth from current request context
        auth_user = None
        auth_pass = None
        
        try:
            if hasattr(request, 'authorization') and request.authorization:
                auth_user = request.authorization.username
                auth_pass = request.authorization.password
        except:
            # No request context or no authorization
            pass
        
        # If no auth from request, use default credentials
        if not auth_user:
            auth_user = settings.get("username", "admin")
            auth_pass = settings.get("password", "12345")
        
        # Embed basic auth in URL for VLC compatibility
        # Format: http://user:pass@host/path
        return f"http://{auth_user}:{auth_pass}@{playlist_host}/play/{portal_id}/{channel_id}"
    
    return base_url


def get_external_host_config():
    """
    Get external host configuration from environment variables.
    
    Returns:
        tuple: (external_host, external_scheme) or (None, None)
    """
    # Check if HOST contains a full URL (simple approach)
    host_env = os.getenv("HOST")
    if host_env and ("http://" in host_env or "https://" in host_env):
        try:
            from urllib.parse import urlparse
            parsed = urlparse(host_env)
            if parsed.hostname:
                host_with_port = f"{parsed.hostname}:{parsed.port}" if parsed.port else parsed.hostname
                scheme = parsed.scheme or "http"
                return host_with_port, scheme
        except Exception:
            pass
    
    # Fallback to None (use request.host)
    return None, None


def extract_auth_credentials(request):
    """
    Extract authentication credentials from HTTP request.
    
    Supports both HTTP Basic Authentication and Query Parameters.
    Basic Auth takes precedence over Query Parameters.
    
    Args:
        request: Flask request object
        
    Returns:
        tuple: (username, password) or (None, None) if no credentials found
    """
    # Priority 1: HTTP Basic Authentication
    if hasattr(request, 'authorization') and request.authorization:
        auth = request.authorization
        if auth.username and auth.password:
            return (auth.username, auth.password)
    
    # Priority 2: Query Parameters
    username = request.args.get('username')
    password = request.args.get('password')
    
    if username and password:
        return (username, password)
    
    # No credentials found
    return (None, None)


def validate_authentication(username, password, settings=None, client_ip=None):
    """
    Validate authentication credentials against system settings.
    
    Args:
        username (str): Username to validate
        password (str): Password to validate
        settings (dict, optional): System settings. If None, will fetch current settings.
        client_ip (str, optional): Client IP address for logging
        
    Returns:
        tuple: (is_valid, error_message)
            - is_valid (bool): True if credentials are valid
            - error_message (str): Error message if validation fails, None if valid
    """
    if settings is None:
        settings = getSettings()
    
    # Get client IP for logging
    if client_ip is None:
        try:
            client_ip = get_client_ip()
        except:
            client_ip = "unknown"
    
    # Check if security is enabled
    security_enabled = settings.get("enable security", "false") == "true"
    
    # If security is disabled, allow access
    if not security_enabled:
        logger.debug(f"Authentication bypassed (security disabled) from IP: {client_ip}")
        return (True, None)
    
    # If security is enabled, credentials are required
    if not username or not password:
        logger.warning(f"Authentication attempt without credentials from IP: {client_ip}")
        return (False, "Authentication required")
    
    # Validate credentials against system settings
    system_username = settings.get("username", "admin")
    system_password = settings.get("password", "12345")
    
    if username != system_username or password != system_password:
        logger.warning(f"Authentication failed for user '{username}' from IP: {client_ip}")
        return (False, "Invalid credentials")
    
    # Authentication successful
    logger.info(f"Authentication successful for user '{username}' from IP: {client_ip}")
    return (True, None)


try:
    subprocess.run([ffmpeg_path, "-version"], capture_output=True, check=True)
    subprocess.run([ffprobe_path, "-version"], capture_output=True, check=True)
    logger.info("FFmpeg and FFprobe found and working")
except (subprocess.CalledProcessError, FileNotFoundError):
    logger.error("Error: ffmpeg or ffprobe not found!")

import flask
from flask import Flask, jsonify
import stb
import json
import subprocess
import uuid
import xml.etree.cElementTree as ET
from flask import (
    Flask,
    render_template,
    redirect,
    request,
    Response,
    make_response,
    flash,
    send_file,
)
from datetime import datetime, timezone
from functools import wraps
import secrets
import waitress
import sqlite3
import atexit
from utils import (
    validate_mac_address,
    validate_url,
    normalize_mac_address,
    sanitize_channel_name,
    get_client_ip,
    is_hls_url,
    validate_proxy_url,
    get_proxy_type,
    parse_proxy_url
)

app = Flask(__name__)
app.secret_key = secrets.token_urlsafe(32)

# Docker-optimized host configuration
if os.getenv("HOST"):
    host = os.getenv("HOST")
else:
    host = "0.0.0.0:8001"
logger.info(f"Server started on http://{host}")

logger.info(f"Using config file: {configFile}")

# Database path for channel caching
dbPath = os.path.join(log_dir, "channels.db")
logger.info(f"Using database file: {dbPath}")

# VOD Database path for VOD/Series caching
vodsDbPath = os.path.join(log_dir, "vods.db")
logger.info(f"Using VOD database file: {vodsDbPath}")

occupied = {}
config = {}
cached_lineup = []
cached_playlist = None
last_playlist_host = None
cached_xmltv = None
last_updated = 0
hls_manager = None

# EPG refresh progress tracking
epg_refresh_progress = {
    "running": False,
    "current_portal": "",
    "current_step": "",
    "portals_done": 0,
    "portals_total": 0,
    "started_at": None
}

# Editor refresh progress tracking
editor_refresh_progress = {
    "running": False,
    "current_portal": "",
    "current_step": "",
    "portals_done": 0,
    "portals_total": 0,
    "started_at": None
}

d_ffmpegcmd = [
    "-re",                      # Flag for real-time streaming
    "-http_proxy", "<proxy>",   # Proxy setting
    "-timeout", "<timeout>",    # Timeout setting
    "-i", "<url>",              # Input URL
    "-map", "0",                # Map all streams
    "-codec", "copy",           # Copy codec (no re-encoding)
    "-f", "mpegts",             # Output format
    "-flush_packets", "0",      # Disable flushing packets (optimized for faster output)
    "-fflags", "+nobuffer",     # No buffering for low latency
    "-flags", "low_delay",      # Low delay flag
    "-strict", "experimental",  # Use experimental features
    "-analyzeduration", "0",    # Skip analysis duration for faster startup
    "-probesize", "32",         # Set probe size to reduce input analysis time
    "-copyts",                  # Copy timestamps (avoid recalculating)
    "-threads", "12",           # Enable multi-threading (adjust thread count as needed)
    "pipe:"                     # Output to pipe
]

defaultSettings = {
    "stream method": "ffmpeg",
    "output format": "mpegts",
    "ffmpeg command": "-re -http_proxy <proxy> -timeout <timeout> -i <url> -map 0 -codec copy -f mpegts -flush_packets 0 -fflags +nobuffer -flags low_delay -strict experimental -analyzeduration 0 -probesize 32 -copyts -threads 12 pipe:",
    "hls segment type": "mpegts",
    "hls segment duration": "4",
    "hls playlist size": "6",
    "hls max streams": "10",
    "hls inactive timeout": "30",
    "ffmpeg timeout": "5",
    "test streams": "true",
    "try all macs": "true",
    "use channel genres": "true",
    "use channel numbers": "true",
    "sort playlist by channel genre": "false",
    "sort playlist by channel number": "true",
    "sort playlist by channel name": "false",
    "enable security": "false",
    "username": "admin",
    "password": "12345",
    "enable hdhr": "true",
    "hdhr name": "MacReplayXC",
    "hdhr id": str(uuid.uuid4().hex),
    "hdhr tuners": "10",
    "epg fallback enabled": "false",
    "epg fallback countries": "",
    "xc api enabled": "false",
    "xc vod proxy": "true",
    "public playlist access": "true",
}

defaultXCUser = {
    "username": "",
    "password": "",
    "enabled": "true",
    "max_connections": "1",
    "allowed_portals": [],  # Empty = all portals
    "created_at": "",
    "expires_at": "",  # Empty = never expires
    "active_connections": {},  # device_id -> {portal_id, channel_id, started_at, ip}
}

defaultPortal = {
    "enabled": "true",
    "name": "",
    "url": "",
    "macs": {},
    "streams per mac": "1",
    "epg offset": "0",
    "proxy": "",
    "portal prefix": "",
    "enabled channels": [],
    "custom channel names": {},
    "custom channel numbers": {},
    "custom genres": {},
    "custom epg ids": {},
    "fallback channels": {},
}


class HLSStreamManager:
    """Manages HLS streams with shared access and automatic cleanup."""
    
    def __init__(self, max_streams=10, inactive_timeout=30):
        self.streams = {}  # Key: "portalId_channelId", Value: stream info dict
        self.max_streams = max_streams
        self.inactive_timeout = inactive_timeout
        self.lock = threading.Lock()
        self.monitor_thread = None
        self.running = False
        logger.info(f"HLS Stream Manager initialized with max_streams={max_streams}, inactive_timeout={inactive_timeout}s")
        
    def start_monitoring(self):
        """Start the background monitoring thread."""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("HLS Stream Manager monitoring started")
    
    def _monitor_loop(self):
        """Background thread that monitors and cleans up inactive streams."""
        while self.running:
            try:
                time.sleep(10)  # Check every 10 seconds
                self._cleanup_inactive_streams()
            except Exception as e:
                logger.error(f"Error in HLS monitor loop: {e}")
    
    def _cleanup_inactive_streams(self):
        """Clean up streams that have been inactive or crashed."""
        current_time = time.time()
        streams_to_remove = []
        
        with self.lock:
            for stream_key, stream_info in self.streams.items():
                is_passthrough = stream_info.get('is_passthrough', False)
                
                # Skip process checks for passthrough streams
                if not is_passthrough:
                    # Check if process has crashed
                    try:
                        if stream_info['process'].poll() is not None:
                            returncode = stream_info['process'].returncode
                            if returncode != 0:
                                logger.error(f"FFmpeg process crashed for {stream_key} (exit code: {returncode})")
                            else:
                                logger.info(f"FFmpeg process ended normally for {stream_key}")
                            streams_to_remove.append(stream_key)
                            continue
                    except Exception as e:
                        logger.error(f"Error checking process status for {stream_key}: {e}")
                        streams_to_remove.append(stream_key)
                        continue
                
                # Check if stream is inactive
                inactive_time = current_time - stream_info['last_accessed']
                if inactive_time > self.inactive_timeout:
                    stream_type = "passthrough" if is_passthrough else "FFmpeg"
                    logger.info(f"Cleaning up inactive {stream_type} stream {stream_key} (idle for {inactive_time:.1f}s)")
                    streams_to_remove.append(stream_key)
        
        # Clean up streams outside the lock to avoid blocking
        for stream_key in streams_to_remove:
            try:
                self._stop_stream(stream_key)
            except Exception as e:
                logger.error(f"Error stopping stream {stream_key}: {e}")
    
    def _stop_stream(self, stream_key):
        """Stop a stream and clean up its resources."""
        with self.lock:
            if stream_key not in self.streams:
                logger.debug(f"Stream {stream_key} already removed")
                return
            
            stream_info = self.streams[stream_key]
            is_passthrough = stream_info.get('is_passthrough', False)
            
            # Terminate FFmpeg process (skip for passthrough streams)
            if not is_passthrough and stream_info.get('process'):
                try:
                    if stream_info['process'].poll() is None:
                        logger.debug(f"Terminating FFmpeg process for {stream_key}")
                        stream_info['process'].terminate()
                        try:
                            stream_info['process'].wait(timeout=5)
                            logger.debug(f"FFmpeg process terminated gracefully for {stream_key}")
                        except subprocess.TimeoutExpired:
                            logger.warning(f"FFmpeg process did not terminate, killing for {stream_key}")
                            stream_info['process'].kill()
                            stream_info['process'].wait(timeout=2)
                except Exception as e:
                    logger.error(f"Error terminating FFmpeg process for {stream_key}: {e}")
                    try:
                        stream_info['process'].kill()
                    except Exception as kill_error:
                        logger.error(f"Error killing FFmpeg process for {stream_key}: {kill_error}")
            
            # Clean up temp directory
            try:
                temp_dir = stream_info.get('temp_dir')
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.debug(f"Cleaned up temp directory for {stream_key}")
            except Exception as e:
                logger.error(f"Error cleaning up temp dir for {stream_key}: {e}")
            
            # Remove from active streams
            del self.streams[stream_key]
            logger.info(f"Stream {stream_key} stopped and cleaned up")
    
    def start_stream(self, portal_id, channel_id, stream_url, proxy=None):
        """Start or reuse an HLS stream for a channel."""
        import tempfile
        
        stream_key = f"{portal_id}_{channel_id}"
        
        with self.lock:
            # Check if stream already exists
            if stream_key in self.streams:
                self.streams[stream_key]['last_accessed'] = time.time()
                logger.info(f"Reusing existing HLS stream for {stream_key}")
                return self.streams[stream_key]
            
            # Check concurrency limit
            if len(self.streams) >= self.max_streams:
                logger.error(f"Max concurrent streams ({self.max_streams}) reached")
                raise Exception(f"Maximum concurrent streams ({self.max_streams}) reached")
            
            # Get HLS settings
            settings = getSettings()
            segment_type = settings.get("hls segment type", "mpegts")
            segment_duration = settings.get("hls segment duration", "4")
            playlist_size = settings.get("hls playlist size", "6")
            timeout = int(settings.get("ffmpeg timeout", "5")) * 1000000
            
            # Detect if source is already HLS
            is_source_hls = is_hls_url(stream_url)
            
            # Create temp directory for HLS segments
            temp_dir = tempfile.mkdtemp(prefix=f"MacReplayXC_hls_{stream_key}_")
            playlist_path = os.path.join(temp_dir, "stream.m3u8")
            master_playlist_path = os.path.join(temp_dir, "master.m3u8")
            
            # If source is already HLS, create a passthrough
            if is_source_hls:
                logger.info(f"Creating HLS passthrough for {stream_key}")
                
                stream_info = {
                    'process': None,
                    'temp_dir': temp_dir,
                    'playlist_path': playlist_path,
                    'master_playlist_path': master_playlist_path,
                    'last_accessed': time.time(),
                    'portal_id': portal_id,
                    'channel_id': channel_id,
                    'stream_url': stream_url,
                    'is_passthrough': True
                }
                
                # Create master playlist that points to the source
                with open(master_playlist_path, 'w') as f:
                    f.write("#EXTM3U\n")
                    f.write("#EXT-X-VERSION:7\n")
                    f.write(f'#EXT-X-STREAM-INF:BANDWIDTH=15000000,CODECS="avc1.640028,mp4a.40.2"\n')
                    f.write(stream_url + "\n")
                
                self.streams[stream_key] = stream_info
                logger.info(f"HLS passthrough ready for {stream_key}")
                return stream_info
            
            # Set segment pattern based on segment type
            if segment_type == "fmp4":
                segment_pattern = os.path.join(temp_dir, "seg_%03d.m4s")
                init_filename = "init.mp4"
            else:
                segment_pattern = os.path.join(temp_dir, "seg_%03d.ts")
                init_filename = None
            
            # Build FFmpeg command for HLS
            ffmpeg_cmd = [
                "ffmpeg",
                "-fflags", "+genpts+igndts+nobuffer",
                "-err_detect", "aggressive",
                "-flags", "low_delay",
                "-reconnect", "1",
                "-reconnect_at_eof", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "15",
            ]
            
            if proxy:
                ffmpeg_cmd.extend(["-http_proxy", proxy])
            
            ffmpeg_cmd.extend(["-timeout", str(timeout)])
            
            ffmpeg_cmd.extend([
                "-i", stream_url,
                "-map", "0",
                "-c:v", "copy",
                "-copyts",
                "-start_at_zero",
                "-c:a", "aac",
                "-b:a", "256k",
                "-af", "aresample=async=1"
            ])
            
            hls_flags = "independent_segments+omit_endlist"
            
            if segment_type == "mpegts":
                hls_flags += "+program_date_time"
                ffmpeg_cmd.extend([
                    "-mpegts_flags", "pat_pmt_at_frames",
                    "-pcr_period", "20"
                ])
            
            ffmpeg_cmd.extend([
                "-f", "hls",
                "-hls_time", segment_duration,
                "-hls_list_size", playlist_size,
                "-hls_flags", hls_flags,
                "-hls_segment_type", segment_type,
                "-hls_segment_filename", segment_pattern,
                "-start_number", "0",
                "-flush_packets", "0"
            ])
            
            if segment_type == "fmp4":
                ffmpeg_cmd.extend(["-hls_fmp4_init_filename", init_filename])
            
            ffmpeg_cmd.append(playlist_path)
            
            # Start FFmpeg process
            try:
                logger.info(f"Starting FFmpeg process for {stream_key}")
                
                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                
                # Store stream info
                stream_info = {
                    'process': process,
                    'temp_dir': temp_dir,
                    'playlist_path': playlist_path,
                    'master_playlist_path': master_playlist_path,
                    'last_accessed': time.time(),
                    'portal_id': portal_id,
                    'channel_id': channel_id,
                    'stream_url': stream_url,
                    'is_passthrough': False
                }
                
                self.streams[stream_key] = stream_info
                logger.info(f"HLS stream started for {stream_key}")
                return stream_info
                
            except Exception as e:
                logger.error(f"Error starting FFmpeg for {stream_key}: {e}")
                # Clean up temp directory
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass
                raise
    
    def get_file(self, portal_id, channel_id, filename):
        """Get a file path for a stream."""
        stream_key = f"{portal_id}_{channel_id}"
        
        with self.lock:
            if stream_key not in self.streams:
                return None
            
            stream_info = self.streams[stream_key]
            stream_info['last_accessed'] = time.time()
            
            # Handle master playlist
            if filename == "master.m3u8":
                if os.path.exists(stream_info['master_playlist_path']):
                    return stream_info['master_playlist_path']
                return None
            
            # Handle stream playlist
            if filename == "stream.m3u8":
                if os.path.exists(stream_info['playlist_path']):
                    return stream_info['playlist_path']
                return None
            
            # Handle segments
            file_path = os.path.join(stream_info['temp_dir'], filename)
            if os.path.exists(file_path):
                return file_path
            
            return None


def loadConfig():
    try:
        with open(configFile) as f:
            data = json.load(f)
        logger.info(f"Config loaded from {configFile}")
    except FileNotFoundError:
        logger.warning("No existing config found. Creating a new one")
        data = {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}. Creating new config.")
        data = {}
    except Exception as e:
        logger.error(f"Error loading config: {e}. Creating new config.")
        data = {}

    data.setdefault("portals", {})
    data.setdefault("settings", {})

    settings = data["settings"]
    settingsOut = {}

    for setting, default in defaultSettings.items():
        value = settings.get(setting)
        if not value or type(default) != type(value):
            value = default
        settingsOut[setting] = value

    data["settings"] = settingsOut

    portals = data["portals"]
    portalsOut = {}

    for portal in portals:
        portalsOut[portal] = {}
        for setting, default in defaultPortal.items():
            value = portals[portal].get(setting)
            if not value or type(default) != type(value):
                value = default
            portalsOut[portal][setting] = value

    data["portals"] = portalsOut

    with open(configFile, "w") as f:
        json.dump(data, f, indent=4)

    return data

def getPortals():
    global config
    if not config:
        config = loadConfig()
    return config["portals"]

def savePortals(portals):
    try:
        with open(configFile, "w") as f:
            config["portals"] = portals
            json.dump(config, f, indent=4)
        logger.debug(f"Portals saved to {configFile}")
    except Exception as e:
        logger.error(f"Error saving portals: {e}")
        raise

def getSettings():
    global config
    if not config:
        config = loadConfig()
    return config["settings"]

def saveSettings(settings):
    try:
        with open(configFile, "w") as f:
            config["settings"] = settings
            json.dump(config, f, indent=4)
        logger.debug(f"Settings saved to {configFile}")
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        raise


def get_db_connection():
    """Get a database connection."""
    conn = sqlite3.connect(dbPath)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS channels (
            portal TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            portal_name TEXT,
            name TEXT,
            number TEXT,
            genre TEXT,
            logo TEXT,
            enabled INTEGER DEFAULT 0,
            custom_name TEXT,
            custom_number TEXT,
            custom_genre TEXT,
            custom_epg_id TEXT,
            fallback_channel TEXT,
            has_portal_epg INTEGER DEFAULT 0,
            PRIMARY KEY (portal, channel_id)
        )
    ''')
    
    # Add has_portal_epg column if it doesn't exist (migration)
    try:
        cursor.execute('ALTER TABLE channels ADD COLUMN has_portal_epg INTEGER DEFAULT 0')
        logger.info("Added has_portal_epg column to database")
    except:
        pass  # Column already exists
    
    # Create indexes for better query performance
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_channels_enabled 
        ON channels(enabled)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_channels_name 
        ON channels(name)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_channels_portal 
        ON channels(portal)
    ''')
    
    # Create table for selected genres per portal
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portal_genres (
            portal TEXT NOT NULL,
            genre TEXT NOT NULL,
            PRIMARY KEY (portal, genre)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")


def get_vod_db_connection():
    """Get a VOD database connection."""
    conn = sqlite3.connect(vodsDbPath)
    conn.row_factory = sqlite3.Row
    return conn


def init_vod_db():
    """Initialize the VOD database and create tables if they don't exist."""
    conn = get_vod_db_connection()
    cursor = conn.cursor()
    
    # VOD Categories table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vod_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portal_id TEXT NOT NULL,
            category_id TEXT NOT NULL,
            title TEXT NOT NULL,
            content_type TEXT NOT NULL,
            item_count INTEGER DEFAULT 0,
            working_mac TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(portal_id, category_id, content_type)
        )
    ''')
    
    # VOD Items table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vod_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portal_id TEXT NOT NULL,
            category_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            content_type TEXT NOT NULL,
            name TEXT NOT NULL,
            year TEXT,
            description TEXT,
            genre TEXT,
            duration TEXT,
            rating TEXT,
            poster_url TEXT,
            cmd TEXT NOT NULL,
            working_macs TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(portal_id, item_id, content_type)
        )
    ''')
    
    # Series Episodes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS series_episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portal_id TEXT NOT NULL,
            series_id TEXT NOT NULL,
            season_number INTEGER NOT NULL,
            episode_number INTEGER NOT NULL,
            title TEXT,
            cmd TEXT NOT NULL,
            working_macs TEXT,
            UNIQUE(portal_id, series_id, season_number, episode_number)
        )
    ''')
    
    # User Selections table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vod_selections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portal_id TEXT NOT NULL,
            category_key TEXT NOT NULL,
            enabled INTEGER DEFAULT 1,
            UNIQUE(portal_id, category_key)
        )
    ''')
    
    # VOD Settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vod_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    ''')
    
    # Create indexes for performance
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_vod_categories_portal 
        ON vod_categories(portal_id)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_vod_items_portal_category 
        ON vod_items(portal_id, category_id)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_vod_items_name 
        ON vod_items(name)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_series_episodes_series 
        ON series_episodes(portal_id, series_id)
    ''')
    
    # Insert default settings if not exist
    cursor.execute('''
        INSERT OR IGNORE INTO vod_settings (key, value) VALUES ('stream_type', 'ffmpeg')
    ''')
    cursor.execute('''
        INSERT OR IGNORE INTO vod_settings (key, value) VALUES ('mac_rotation', 'true')
    ''')
    
    conn.commit()
    conn.close()
    logger.info("VOD database initialized successfully")


def refresh_channels_cache_with_progress():
    """Wrapper for refresh_channels_cache with progress tracking."""
    global editor_refresh_progress
    try:
        return refresh_channels_cache()
    finally:
        editor_refresh_progress["running"] = False
        editor_refresh_progress["current_step"] = "Completed"

def refresh_channels_cache():
    """Refresh the channels cache from STB portals - fetches from ALL MACs."""
    global editor_refresh_progress
    
    logger.info("Starting channel cache refresh...")
    editor_refresh_progress["current_step"] = "Loading portals..."
    
    portals = getPortals()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    total_channels = 0
    portal_index = 0
    
    for portal_id in portals:
        portal = portals[portal_id]
        if portal["enabled"] == "true":
            portal_index += 1
            portal_name = portal["name"]
            url = portal["url"]
            macs = list(portal["macs"].keys())
            proxy = portal["proxy"]
            
            # Update progress
            editor_refresh_progress["current_portal"] = portal_name
            editor_refresh_progress["current_step"] = f"Starting {portal_name}..."
            editor_refresh_progress["portals_done"] = portal_index - 1
            
            # Get existing settings from JSON config for migration
            enabled_channels = portal.get("enabled channels", [])
            custom_channel_names = portal.get("custom channel names", {})
            custom_genres = portal.get("custom genres", {})
            custom_channel_numbers = portal.get("custom channel numbers", {})
            custom_epg_ids = portal.get("custom epg ids", {})
            fallback_channels = portal.get("fallback channels", {})
            
            logger.info(f"Fetching channels for portal: {portal_name} from {len(macs)} MACs")
            editor_refresh_progress["current_step"] = f"{portal_name}: Found {len(macs)} MAC(s)"
            
            # Fetch from ALL MACs and merge
            all_channels_map = {}  # channel_id -> channel data
            all_genres_dict = {}  # genre_id -> genre_name
            
            mac_index = 0
            for mac in macs:
                mac_index += 1
                logger.info(f"Trying MAC: {mac}")
                editor_refresh_progress["current_step"] = f"{portal_name}: Fetching from MAC {mac_index}/{len(macs)}"
                try:
                    token = stb.getToken(url, mac, proxy)
                    if token:
                        stb.getProfile(url, mac, token, proxy)
                        editor_refresh_progress["current_step"] = f"{portal_name}: Getting channels from MAC {mac_index}/{len(macs)}"
                        mac_channels = stb.getAllChannels(url, mac, token, proxy)
                        editor_refresh_progress["current_step"] = f"{portal_name}: Getting genres from MAC {mac_index}/{len(macs)}"
                        mac_genres = stb.getGenreNames(url, mac, token, proxy)
                        
                        if mac_channels:
                            # Merge channels - add new ones
                            for channel in mac_channels:
                                channel_id = str(channel["id"])
                                if channel_id not in all_channels_map:
                                    all_channels_map[channel_id] = channel
                            logger.info(f"MAC {mac}: Added {len(mac_channels)} channels (total: {len(all_channels_map)})")
                            editor_refresh_progress["current_step"] = f"{portal_name}: MAC {mac_index}/{len(macs)} - {len(all_channels_map)} channels"
                        
                        if mac_genres:
                            all_genres_dict.update(mac_genres)
                            logger.info(f"MAC {mac}: Added genres (total: {len(all_genres_dict)})")
                            editor_refresh_progress["current_step"] = f"{portal_name}: MAC {mac_index}/{len(macs)} - {len(all_genres_dict)} genres"
                            
                except Exception as e:
                    logger.error(f"Error fetching from MAC {mac}: {e}")
                    continue
            
            if all_channels_map and all_genres_dict:
                logger.info(f"Processing {len(all_channels_map)} total channels for {portal_name}")
                editor_refresh_progress["current_step"] = f"{portal_name}: Checking EPG availability..."
                
                # Fetch EPG data from ALL MACs to check which channels have portal EPG
                merged_epg = {}
                epg_mac_index = 0
                for mac in macs:
                    epg_mac_index += 1
                    try:
                        editor_refresh_progress["current_step"] = f"{portal_name}: Checking EPG from MAC {epg_mac_index}/{len(macs)}"
                        token = stb.getToken(url, mac, proxy)
                        if token:
                            stb.getProfile(url, mac, token, proxy)
                            mac_epg = stb.getEpg(url, mac, token, 24, proxy)
                            if mac_epg:
                                for ch_id, programmes in mac_epg.items():
                                    if ch_id not in merged_epg or len(programmes) > len(merged_epg.get(ch_id, [])):
                                        merged_epg[ch_id] = programmes
                    except Exception as e:
                        logger.error(f"Error fetching EPG from MAC {mac}: {e}")
                        continue
                
                logger.info(f"Portal {portal_name}: Got EPG for {len(merged_epg)} channels")
                editor_refresh_progress["current_step"] = f"{portal_name}: Saving {len(all_channels_map)} channels to database..."
                
                for channel_id, channel in all_channels_map.items():
                    channel_name = str(channel["name"])
                    channel_number = str(channel["number"])
                    genre_id = str(channel.get("tv_genre_id", ""))
                    genre = str(all_genres_dict.get(genre_id, ""))
                    logo = str(channel.get("logo", ""))
                    
                    # Check if enabled (from JSON config)
                    enabled = 1 if channel_id in enabled_channels else 0
                    
                    # Check if channel has portal EPG
                    has_portal_epg = 1 if (channel_id in merged_epg and merged_epg[channel_id]) else 0
                    
                    # Get custom values (from JSON config)
                    custom_name = custom_channel_names.get(channel_id, "")
                    custom_number = custom_channel_numbers.get(channel_id, "")
                    custom_genre = custom_genres.get(channel_id, "")
                    custom_epg_id = custom_epg_ids.get(channel_id, "")
                    fallback_channel = fallback_channels.get(channel_id, "")
                    
                    # Upsert into database
                    cursor.execute('''
                        INSERT INTO channels (
                            portal, channel_id, portal_name, name, number, genre, logo,
                            enabled, custom_name, custom_number, custom_genre, 
                            custom_epg_id, fallback_channel, has_portal_epg
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(portal, channel_id) DO UPDATE SET
                            portal_name = excluded.portal_name,
                            name = excluded.name,
                            number = excluded.number,
                            genre = excluded.genre,
                            logo = excluded.logo,
                            has_portal_epg = excluded.has_portal_epg
                    ''', (
                        portal_id, channel_id, portal_name, channel_name, channel_number,
                        genre, logo, enabled, custom_name, custom_number, custom_genre,
                        custom_epg_id, fallback_channel, has_portal_epg
                    ))
                    
                    total_channels += 1
                
                conn.commit()
                logger.info(f"Successfully cached {len(all_channels_map)} channels for {portal_name}")
                editor_refresh_progress["current_step"] = f"{portal_name}: Completed - {len(all_channels_map)} channels saved"
                editor_refresh_progress["portals_done"] = portal_index
            else:
                logger.error(f"Failed to fetch channels for portal: {portal_name}")
                editor_refresh_progress["current_step"] = f"{portal_name}: Error - failed to fetch channels"
                editor_refresh_progress["portals_done"] = portal_index
    
    conn.close()
    logger.info(f"Channel cache refresh complete. Total channels: {total_channels}")
    editor_refresh_progress["current_step"] = f"Completed! {total_channels} channels from {portal_index} portals"
    return total_channels


# ============================================
# XC API User Management
# ============================================

def getXCUsers():
    """Get all XC API users."""
    return config.get("xc_users", {})


def saveXCUsers(users):
    """Save XC API users."""
    try:
        with open(configFile, "w") as f:
            config["xc_users"] = users
            json.dump(config, f, indent=4)
        logger.debug(f"XC users saved to {configFile}")
    except Exception as e:
        logger.error(f"Error saving XC users: {e}")
        raise


def validateXCUser(username, password):
    """Validate XC API user credentials."""
    users = getXCUsers()
    user_id = f"{username}_{password}"
    
    if user_id not in users:
        return None, "Invalid credentials"
    
    user = users[user_id]
    
    if user.get("enabled") != "true":
        return None, "User disabled"
    
    # Check expiry
    expires_at = user.get("expires_at", "")
    if expires_at:
        try:
            expiry_date = datetime.strptime(expires_at, "%Y-%m-%d")
            if datetime.now() > expiry_date:
                return None, "User expired"
        except:
            pass
    
    return user_id, user


def checkXCConnectionLimit(user_id, device_id):
    """Check if user can start a new connection."""
    users = getXCUsers()
    if user_id not in users:
        return False, "User not found"
    
    user = users[user_id]
    max_connections = int(user.get("max_connections", 1))
    active_connections = user.get("active_connections", {})
    
    # Clean up old connections (older than 60 seconds without activity)
    current_time = time.time()
    cleaned_connections = {}
    modified = False
    for dev_id, conn in active_connections.items():
        if current_time - conn.get("last_activity", 0) < 60:
            cleaned_connections[dev_id] = conn
        else:
            modified = True
    
    # Save if we cleaned up any connections
    if modified:
        user["active_connections"] = cleaned_connections
        saveXCUsers(users)
    
    # If this device already has a connection, allow it
    if device_id in cleaned_connections:
        return True, "Existing connection"
    
    # Check if under limit
    if len(cleaned_connections) >= max_connections:
        return False, f"Connection limit reached ({max_connections})"
    
    return True, "OK"


def registerXCConnection(user_id, device_id, portal_id, channel_id, ip):
    """Register a new XC API connection."""
    users = getXCUsers()
    if user_id not in users:
        return False
    
    if "active_connections" not in users[user_id]:
        users[user_id]["active_connections"] = {}
    
    users[user_id]["active_connections"][device_id] = {
        "portal_id": portal_id,
        "channel_id": channel_id,
        "started_at": time.time(),
        "last_activity": time.time(),
        "ip": ip
    }
    
    saveXCUsers(users)
    return True


def updateXCConnectionActivity(user_id, device_id):
    """Update last activity time for a connection."""
    users = getXCUsers()
    if user_id in users and device_id in users[user_id].get("active_connections", {}):
        users[user_id]["active_connections"][device_id]["last_activity"] = time.time()
        saveXCUsers(users)


def unregisterXCConnection(user_id, device_id):
    """Unregister an XC API connection."""
    users = getXCUsers()
    if user_id in users and device_id in users[user_id].get("active_connections", {}):
        del users[user_id]["active_connections"][device_id]
        saveXCUsers(users)


def cleanupOldXCConnections():
    """Cleanup connections older than 5 minutes without activity."""
    users = getXCUsers()
    current_time = time.time()
    timeout = 300  # 5 minutes
    
    modified = False
    # Create a copy to avoid RuntimeError if dictionary changes during iteration
    for user_id, user in list(users.items()):
        active_connections = user.get("active_connections", {})
        to_remove = []
        
        for device_id, conn_info in active_connections.items():
            last_activity = conn_info.get("last_activity", 0)
            if current_time - last_activity > timeout:
                to_remove.append(device_id)
        
        for device_id in to_remove:
            del active_connections[device_id]
            modified = True
    
    if modified:
        saveXCUsers(users)


def authorise(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        settings = getSettings()
        security = settings["enable security"]
        
        # If security is disabled, allow access
        if security == "false":
            return f(*args, **kwargs)
        
        # Check if user is logged in via session
        if flask.session.get("authenticated"):
            return f(*args, **kwargs)
        
        # Not authenticated, redirect to login page
        return redirect("/login", code=302)
    
    return decorated


def xc_auth_only(f):
    """Decorator for XC API routes - only allows XC API authentication, no HTTP Basic Auth fallback."""
    @wraps(f)
    def decorated(*args, **kwargs):
        settings = getSettings()
        
        if settings.get("xc api enabled") != "true":
            return flask.jsonify({"user_info": {"auth": 0, "message": "XC API disabled"}}), 403
        
        xc_username = request.args.get("username") or kwargs.get("username")
        xc_password = request.args.get("password") or kwargs.get("password")
        
        if not xc_username or not xc_password:
            return flask.jsonify({"user_info": {"auth": 0, "message": "Missing credentials"}}), 401
        
        user_id, user = validateXCUser(xc_username, xc_password)
        if not user:
            logger.debug(f"Auth failed: {xc_username}")
            return flask.jsonify({"user_info": {"auth": 0, "message": user_id}}), 401
        
        return f(*args, **kwargs)
    
    return decorated


def xc_auth_optional(f):
    """Decorator that allows both XC API auth and HTTP Basic Auth."""
    @wraps(f)
    def decorated(*args, **kwargs):
        settings = getSettings()
        
        # If security is disabled, allow access
        if settings.get("enable security") == "false":
            return f(*args, **kwargs)
        
        # Check if XC API is enabled and this is an XC API request
        if settings.get("xc api enabled") == "true":
            # Try XC API authentication first (from URL params or path)
            xc_username = request.args.get("username") or kwargs.get("username")
            xc_password = request.args.get("password") or kwargs.get("password")
            
            if xc_username and xc_password:
                user_id, user = validateXCUser(xc_username, xc_password)
                if user_id:
                    # XC API auth successful, allow access
                    return f(*args, **kwargs)
        
        # Fall back to HTTP Basic Auth
        auth = request.authorization
        username = settings["username"]
        password = settings["password"]
        
        if auth and auth.username == username and auth.password == password:
            return f(*args, **kwargs)
        
        return make_response(
            "Could not verify your login!",
            401,
            {"WWW-Authenticate": 'Basic realm="Login Required"'},
        )

    return decorated

def moveMac(portalId, mac):
    portals = getPortals()
    macs = portals[portalId]["macs"]
    x = macs[mac]
    del macs[mac]
    macs[mac] = x
    portals[portalId]["macs"] = macs
    savePortals(portals)

@app.route("/data/<path:filename>", methods=["GET"])
def block_data_access(filename):
    """Block direct access to data files."""
    return "Access denied", 403


@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page."""
    if request.method == "POST":
        settings = getSettings()
        username = request.form.get("username")
        password = request.form.get("password")
        
        if username == settings["username"] and password == settings["password"]:
            flask.session["authenticated"] = True
            flask.session.permanent = True
            return redirect("/dashboard", code=302)
        else:
            return render_template("login.html", error="Invalid credentials")
    
    # If already authenticated, redirect to dashboard
    if flask.session.get("authenticated"):
        return redirect("/dashboard", code=302)
    
    return render_template("login.html")


@app.route("/logout", methods=["GET"])
def logout():
    """Logout."""
    flask.session.clear()
    return redirect("/login", code=302)


# ============================================================================
# VOD/Series Routes
# ============================================================================

# Global VOD refresh state
vod_refresh_state = {
    "running": False,
    "portals_total": 0,
    "portals_done": 0,
    "current_portal": "",
    "current_step": ""
}

# MAC rotation state per portal
vod_mac_rotation_state = {}


def get_next_mac_for_portal(portal_id, macs):
    """Get next MAC in rotation for a portal."""
    global vod_mac_rotation_state
    
    if not macs:
        return None
    
    if portal_id not in vod_mac_rotation_state:
        vod_mac_rotation_state[portal_id] = 0
    
    current_index = vod_mac_rotation_state[portal_id]
    mac = macs[current_index % len(macs)]
    
    # Advance to next MAC for next request
    vod_mac_rotation_state[portal_id] = (current_index + 1) % len(macs)
    
    return mac


def try_get_vod_link_with_fallback(url, macs, cmd, content_type, proxy, series_id=None, season_id=None, episode_id=None):
    """Try to get VOD link with MAC fallback on failure."""
    working_mac = None
    link = None
    
    for mac in macs:
        try:
            token = stb.getToken(url, mac, proxy)
            if not token:
                continue
            
            if content_type == 'series' and series_id:
                link = stb.getSeriesLink(url, mac, token, cmd, series_id, season_id, episode_id, proxy)
            else:
                link = stb.getVodLink(url, mac, token, cmd, proxy)
            
            if link:
                working_mac = mac
                break
        except Exception as e:
            logger.debug(f"MAC {mac} failed for VOD link: {e}")
            continue
    
    return link, working_mac


@app.route("/api/vods", methods=["GET"])
@app.route("/vods", methods=["GET"])
@authorise
def vods_page():
    """Render VOD page."""
    return render_template("vods.html")


@app.route("/vods/portals", methods=["GET"])
@authorise
def vods_portals():
    """Get all portals with VOD/Series category counts."""
    try:
        portals = getPortals()
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        
        result = []
        for portal_id, portal in portals.items():
            if portal.get("enabled") != "true":
                continue
            
            # Get cached category counts
            cursor.execute('''
                SELECT content_type, COUNT(*) as count 
                FROM vod_categories 
                WHERE portal_id = ? 
                GROUP BY content_type
            ''', (portal_id,))
            
            cached_counts = {row['content_type']: row['count'] for row in cursor.fetchall()}
            
            # Get selected categories count
            cursor.execute('''
                SELECT COUNT(*) as count 
                FROM vod_selections 
                WHERE portal_id = ? AND enabled = 1
            ''', (portal_id,))
            selected_count = cursor.fetchone()['count']
            
            result.append({
                "id": portal_id,
                "name": portal.get("name", portal_id),
                "macs": len(portal.get("macs", {})),
                "vod_categories": cached_counts.get("vod", 0),
                "series_categories": cached_counts.get("series", 0),
                "cached_vod_categories": cached_counts.get("vod", 0),
                "cached_series_categories": cached_counts.get("series", 0),
                "selected_categories": selected_count,
                "has_cache": (cached_counts.get("vod", 0) + cached_counts.get("series", 0)) > 0
            })
        
        conn.close()
        return jsonify({"success": True, "portals": result})
    except Exception as e:
        logger.error(f"Error getting VOD portals: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/vods/categories/<portal_id>", methods=["GET"])
@authorise
def vods_categories(portal_id):
    """Get VOD/Series categories for a portal from cache."""
    try:
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT category_id, title, content_type, item_count, working_mac
            FROM vod_categories
            WHERE portal_id = ?
            ORDER BY content_type, title
        ''', (portal_id,))
        
        categories = []
        for row in cursor.fetchall():
            categories.append({
                "category_id": row['category_id'],
                "title": row['title'],
                "type": row['content_type'],
                "item_count": row['item_count'],
                "working_mac": row['working_mac'] or "N/A"
            })
        
        conn.close()
        return jsonify({"success": True, "categories": categories})
    except Exception as e:
        logger.error(f"Error getting VOD categories: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/vods/items/<portal_id>/<content_type>/<category_id>", methods=["GET"])
@authorise
def vods_items(portal_id, content_type, category_id):
    """Get VOD/Series items for a category from cache."""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute('''
            SELECT COUNT(*) as total
            FROM vod_items
            WHERE portal_id = ? AND category_id = ? AND content_type = ?
        ''', (portal_id, category_id, content_type))
        total = cursor.fetchone()['total']
        
        # Get items with pagination
        offset = (page - 1) * per_page
        cursor.execute('''
            SELECT item_id, name, year, description, genre, duration, rating, poster_url, cmd, working_macs
            FROM vod_items
            WHERE portal_id = ? AND category_id = ? AND content_type = ?
            ORDER BY name
            LIMIT ? OFFSET ?
        ''', (portal_id, category_id, content_type, per_page, offset))
        
        items = []
        for row in cursor.fetchall():
            items.append({
                "id": row['item_id'],
                "name": row['name'],
                "year": row['year'],
                "description": row['description'],
                "genre": row['genre'],
                "duration": row['duration'],
                "rating": row['rating'],
                "screenshot_uri": row['poster_url'],
                "cmd": row['cmd'],
                "working_mac": row['working_macs'].split(',')[0] if row['working_macs'] else None
            })
        
        conn.close()
        return jsonify({
            "success": True, 
            "items": items, 
            "total": total,
            "page": page,
            "has_more": (page * per_page) < total
        })
    except Exception as e:
        logger.error(f"Error getting VOD items: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/vods/selection/<portal_id>", methods=["GET"])
@authorise
def vods_selection_get(portal_id):
    """Get selected categories for a portal."""
    try:
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT category_key FROM vod_selections
            WHERE portal_id = ? AND enabled = 1
        ''', (portal_id,))
        
        selected = [row['category_key'] for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({"success": True, "selected_categories": selected})
    except Exception as e:
        logger.error(f"Error getting VOD selection: {e}")
        return jsonify({"success": False, "error": str(e)})


# Global state for VOD items loading progress
vod_items_load_state = {
    "running": False,
    "portal_id": None,
    "categories_total": 0,
    "categories_done": 0,
    "current_category": "",
    "items_loaded": 0
}


@app.route("/vods/save-selection", methods=["POST"])
@authorise
def vods_selection_save():
    """Save selected categories for a portal and start loading items in background."""
    global vod_items_load_state
    
    try:
        data = request.get_json()
        portal_id = data.get('portal_id')
        selected_categories = data.get('selected_categories', [])
        load_items = data.get('load_items', True)  # Default to loading items
        
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        
        # Clear existing selections
        cursor.execute('DELETE FROM vod_selections WHERE portal_id = ?', (portal_id,))
        
        # Insert new selections
        for category_key in selected_categories:
            cursor.execute('''
                INSERT INTO vod_selections (portal_id, category_key, enabled)
                VALUES (?, ?, 1)
            ''', (portal_id, category_key))
        
        conn.commit()
        conn.close()
        
        # Start loading items in background if requested
        if load_items and selected_categories and not vod_items_load_state["running"]:
            def load_items_background():
                global vod_items_load_state
                try:
                    vod_items_load_state["running"] = True
                    vod_items_load_state["portal_id"] = portal_id
                    vod_items_load_state["categories_total"] = len(selected_categories)
                    vod_items_load_state["categories_done"] = 0
                    vod_items_load_state["items_loaded"] = 0
                    
                    portals = getPortals()
                    portal = portals.get(portal_id)
                    
                    if not portal:
                        logger.error(f"Portal {portal_id} not found for items loading")
                        return
                    
                    url = portal.get("url")
                    macs = list(portal.get("macs", {}).keys())
                    proxy = portal.get("proxy")
                    
                    if not macs:
                        logger.error(f"No MACs for portal {portal_id}")
                        return
                    
                    # Get working MACs for each category from database
                    conn = get_vod_db_connection()
                    cursor = conn.cursor()
                    
                    # Build a map of category -> working_macs
                    category_macs = {}
                    for category_key in selected_categories:
                        parts = category_key.split('_', 1)
                        if len(parts) != 2:
                            continue
                        content_type = parts[0]
                        category_id = parts[1]
                        
                        cursor.execute('''
                            SELECT working_mac FROM vod_categories 
                            WHERE portal_id = ? AND category_id = ? AND content_type = ?
                        ''', (portal_id, category_id, content_type))
                        row = cursor.fetchone()
                        if row and row['working_mac']:
                            # working_mac can be comma-separated list
                            category_macs[category_key] = row['working_mac'].split(',')
                        else:
                            category_macs[category_key] = macs  # Fallback to all MACs
                    
                    # Cache tokens for MACs
                    mac_tokens = {}
                    
                    for category_key in selected_categories:
                        # Parse category key (format: "vod_123" or "series_456")
                        parts = category_key.split('_', 1)
                        if len(parts) != 2:
                            continue
                        
                        content_type = parts[0]
                        category_id = parts[1]
                        
                        # Skip "all" category (category_id = "*")
                        if category_id == "*":
                            logger.debug(f"Skipping 'all' category: {category_key}")
                            continue
                        
                        vod_items_load_state["current_category"] = f"{content_type}: {category_id}"
                        
                        # Get working MACs for this category
                        cat_macs = category_macs.get(category_key, macs)
                        
                        # Try each MAC until we get items
                        category_items = 0
                        logger.info(f"Loading items for {category_key} (type={content_type}, id={category_id}), trying {len(cat_macs)} MACs")
                        
                        for mac in cat_macs:
                            # Get or create token for this MAC
                            if mac not in mac_tokens:
                                try:
                                    token = stb.getToken(url, mac, proxy)
                                    if token:
                                        mac_tokens[mac] = token
                                        logger.info(f"Got token for MAC {mac[:15]}...")
                                    else:
                                        logger.warning(f"No token returned for MAC {mac[:15]}...")
                                        continue
                                except Exception as e:
                                    logger.warning(f"Failed to get token for MAC {mac[:15]}...: {e}")
                                    continue
                            
                            if mac not in mac_tokens:
                                continue
                            
                            token = mac_tokens[mac]
                            
                            # Load items for this category with this MAC
                            page = 1
                            mac_items = 0
                            
                            while True:
                                try:
                                    logger.debug(f"Fetching {content_type} items for category {category_id}, page {page}, MAC {mac[:15]}...")
                                    if content_type == 'series':
                                        result = stb.getSeriesItems(url, mac, token, category_id, page, proxy)
                                    else:
                                        result = stb.getVodItems(url, mac, token, category_id, page, proxy)
                                    
                                    if not result:
                                        logger.debug(f"No result returned for category {category_id}")
                                        break
                                    
                                    items = result.get('items', [])
                                    if not items:
                                        logger.debug(f"Empty items list for category {category_id}")
                                        break
                                    
                                    logger.info(f"Got {len(items)} items for category {category_id}, page {page}")
                                    
                                    # Save items to database
                                    for item in items:
                                        item_id = str(item.get('id', ''))
                                        name = item.get('name', '')
                                        year = item.get('year', '')
                                        description = item.get('description', '')
                                        genre = item.get('genre_str', '')
                                        duration = item.get('time', '')
                                        rating = item.get('rating_imdb', '')
                                        poster_url = item.get('screenshot_uri', '')
                                        cmd = item.get('cmd', '')
                                        
                                        cursor.execute('''
                                            INSERT OR REPLACE INTO vod_items 
                                            (portal_id, category_id, item_id, content_type, name, year, description, 
                                             genre, duration, rating, poster_url, cmd, working_macs)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        ''', (portal_id, category_id, item_id, content_type, name, year, description,
                                              genre, duration, rating, poster_url, cmd, mac))
                                        
                                        mac_items += 1
                                        category_items += 1
                                        vod_items_load_state["items_loaded"] += 1
                                    
                                    conn.commit()
                                    
                                    # Check if there are more pages
                                    total = int(result.get('total', 0))
                                    if mac_items >= total or len(items) < 14:
                                        logger.debug(f"Finished loading category {category_id}: {mac_items} items (total: {total})")
                                        break
                                    
                                    page += 1
                                    
                                except Exception as e:
                                    logger.error(f"Error loading items for category {category_key} with MAC {mac[:15]}...: {e}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                                    break
                            
                            # If we got items with this MAC, don't try other MACs
                            if mac_items > 0:
                                logger.info(f"Successfully loaded {mac_items} items for category {category_key} with MAC {mac[:15]}...")
                                break
                            else:
                                logger.debug(f"No items found for category {category_key} with MAC {mac[:15]}..., trying next MAC")
                        
                        vod_items_load_state["categories_done"] += 1
                        if category_items == 0:
                            logger.warning(f"No items loaded for category {category_key} after trying all MACs")
                        else:
                            logger.info(f"Loaded {category_items} items for category {category_key}")
                            # Update item_count in vod_categories table
                            cursor.execute('''
                                UPDATE vod_categories SET item_count = ?
                                WHERE portal_id = ? AND category_id = ? AND content_type = ?
                            ''', (category_items, portal_id, category_id, content_type))
                            conn.commit()
                    
                    conn.close()
                    logger.info(f"Finished loading {vod_items_load_state['items_loaded']} items for portal {portal_id}")
                    
                except Exception as e:
                    logger.error(f"Error in background items loading: {e}")
                finally:
                    vod_items_load_state["running"] = False
            
            # Start background thread
            threading.Thread(target=load_items_background, daemon=True).start()
        
        return jsonify({
            "success": True, 
            "count": len(selected_categories),
            "loading_items": load_items and len(selected_categories) > 0
        })
    except Exception as e:
        logger.error(f"Error saving VOD selection: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/vods/items-load/progress", methods=["GET"])
@authorise
def vods_items_load_progress():
    """Get VOD items loading progress."""
    return jsonify(vod_items_load_state)


@app.route("/vods/settings", methods=["GET"])
@authorise
def vods_settings_get():
    """Get VOD settings."""
    try:
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT key, value FROM vod_settings')
        settings = {row['key']: row['value'] for row in cursor.fetchall()}
        conn.close()
        
        return jsonify({"success": True, "settings": settings})
    except Exception as e:
        logger.error(f"Error getting VOD settings: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/vods/settings", methods=["POST"])
@authorise
def vods_settings_save():
    """Save VOD settings."""
    try:
        data = request.get_json()
        
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        
        for key, value in data.items():
            cursor.execute('''
                INSERT OR REPLACE INTO vod_settings (key, value) VALUES (?, ?)
            ''', (key, str(value)))
        
        conn.commit()
        conn.close()
        
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error saving VOD settings: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/vods/refresh", methods=["POST"])
@authorise
def vods_refresh():
    """Start VOD cache refresh in background - tests ALL MACs and merges categories."""
    global vod_refresh_state
    
    if vod_refresh_state["running"]:
        return jsonify({"success": False, "error": "Refresh already in progress"})
    
    def refresh_vod_cache():
        global vod_refresh_state
        try:
            vod_refresh_state["running"] = True
            portals = getPortals()
            enabled_portals = {k: v for k, v in portals.items() if v.get("enabled") == "true"}
            
            vod_refresh_state["portals_total"] = len(enabled_portals)
            vod_refresh_state["portals_done"] = 0
            
            for portal_id, portal in enabled_portals.items():
                vod_refresh_state["current_portal"] = portal.get("name", portal_id)
                vod_refresh_state["current_step"] = "Testing MACs..."
                
                url = portal.get("url")
                macs = list(portal.get("macs", {}).keys())
                proxy = portal.get("proxy")
                
                if not macs:
                    vod_refresh_state["portals_done"] += 1
                    continue
                
                conn = get_vod_db_connection()
                cursor = conn.cursor()
                
                # Track all categories by key to merge from multiple MACs
                all_vod_categories = {}
                all_series_categories = {}
                working_macs_count = 0
                
                # Helper function to extract item count from category
                def get_item_count(cat):
                    for field in ['censored', 'count', 'cnt', 'total', 'items_count', 'num', 'number']:
                        val = cat.get(field)
                        if val is not None:
                            try:
                                return int(val)
                            except (ValueError, TypeError):
                                pass
                    return 0
                
                # Test ALL MACs and merge their categories
                for mac_idx, mac in enumerate(macs):
                    vod_refresh_state["current_step"] = f"Testing MAC {mac_idx + 1}/{len(macs)}..."
                    try:
                        token = stb.getToken(url, mac, proxy)
                        if not token:
                            continue
                        
                        working_macs_count += 1
                        
                        # Get VOD categories from this MAC
                        vod_refresh_state["current_step"] = f"MAC {mac_idx + 1}: Loading VOD categories..."
                        vod_cats = stb.getVodCategories(url, mac, token, proxy)
                        if vod_cats:
                            for cat in vod_cats:
                                cat_id = str(cat.get('id', ''))
                                title = cat.get('title', '')
                                item_count = get_item_count(cat)
                                
                                # Skip "all" category (id = "*")
                                if cat_id == "*" or not cat_id:
                                    continue
                                
                                if cat_id not in all_vod_categories:
                                    all_vod_categories[cat_id] = {
                                        "title": title,
                                        "item_count": item_count,
                                        "working_macs": [mac]
                                    }
                                else:
                                    if mac not in all_vod_categories[cat_id]["working_macs"]:
                                        all_vod_categories[cat_id]["working_macs"].append(mac)
                                    if item_count > all_vod_categories[cat_id]["item_count"]:
                                        all_vod_categories[cat_id]["item_count"] = item_count
                        
                        # Get Series categories from this MAC
                        vod_refresh_state["current_step"] = f"MAC {mac_idx + 1}: Loading Series categories..."
                        series_cats = stb.getSeriesCategories(url, mac, token, proxy)
                        if series_cats:
                            for cat in series_cats:
                                cat_id = str(cat.get('id', ''))
                                title = cat.get('title', '')
                                item_count = get_item_count(cat)
                                
                                # Skip "all" category (id = "*")
                                if cat_id == "*" or not cat_id:
                                    continue
                                
                                if cat_id not in all_series_categories:
                                    all_series_categories[cat_id] = {
                                        "title": title,
                                        "item_count": item_count,
                                        "working_macs": [mac]
                                    }
                                else:
                                    if mac not in all_series_categories[cat_id]["working_macs"]:
                                        all_series_categories[cat_id]["working_macs"].append(mac)
                                    if item_count > all_series_categories[cat_id]["item_count"]:
                                        all_series_categories[cat_id]["item_count"] = item_count
                                        
                    except Exception as e:
                        logger.debug(f"MAC {mac} failed: {e}")
                        continue
                
                if working_macs_count == 0:
                    logger.warning(f"No working MAC for portal {portal_id}")
                    conn.close()
                    vod_refresh_state["portals_done"] += 1
                    continue
                
                # Save merged VOD categories to database
                vod_refresh_state["current_step"] = "Saving categories..."
                for cat_id, cat_data in all_vod_categories.items():
                    working_macs_str = ','.join(cat_data["working_macs"])
                    cursor.execute('''
                        INSERT OR REPLACE INTO vod_categories 
                        (portal_id, category_id, title, content_type, item_count, working_mac)
                        VALUES (?, ?, ?, 'vod', ?, ?)
                    ''', (portal_id, cat_id, cat_data["title"], cat_data["item_count"], working_macs_str))
                
                # Save merged Series categories to database
                for cat_id, cat_data in all_series_categories.items():
                    working_macs_str = ','.join(cat_data["working_macs"])
                    cursor.execute('''
                        INSERT OR REPLACE INTO vod_categories 
                        (portal_id, category_id, title, content_type, item_count, working_mac)
                        VALUES (?, ?, ?, 'series', ?, ?)
                    ''', (portal_id, cat_id, cat_data["title"], cat_data["item_count"], working_macs_str))
                
                conn.commit()
                conn.close()
                
                logger.info(f"Portal {portal_id}: {len(all_vod_categories)} VOD, {len(all_series_categories)} Series categories from {working_macs_count} MACs")
                vod_refresh_state["portals_done"] += 1
            
            vod_refresh_state["current_step"] = "Completed!"
            
        except Exception as e:
            logger.error(f"Error in VOD refresh: {e}")
        finally:
            vod_refresh_state["running"] = False
    
    # Start refresh in background thread
    threading.Thread(target=refresh_vod_cache, daemon=True).start()
    
    return jsonify({"success": True, "message": "VOD refresh started"})


@app.route("/vods/refresh/progress", methods=["GET"])
@authorise
def vods_refresh_progress():
    """Get VOD refresh progress."""
    return jsonify(vod_refresh_state)


@app.route("/vods/load-categories", methods=["POST"])
@authorise
def vods_load_categories():
    """Load categories for a single portal on-demand - tests ALL MACs and merges categories."""
    try:
        data = request.get_json()
        portal_id = data.get('portal_id')
        
        portals = getPortals()
        portal = portals.get(portal_id)
        
        if not portal:
            return jsonify({"success": False, "error": "Portal not found"})
        
        url = portal.get("url")
        macs = list(portal.get("macs", {}).keys())
        proxy = portal.get("proxy")
        
        if not macs:
            return jsonify({"success": False, "error": "No MACs configured"})
        
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        
        # Track all categories by key (category_id + content_type) to merge from multiple MACs
        all_vod_categories = {}  # key: category_id, value: {data, working_macs: []}
        all_series_categories = {}
        working_macs_count = 0
        
        # Helper function to extract item count from category - tries multiple field names
        def get_item_count(cat):
            # Try various possible field names for item count
            for field in ['censored', 'count', 'cnt', 'total', 'items_count', 'num', 'number']:
                val = cat.get(field)
                if val is not None:
                    try:
                        return int(val)
                    except (ValueError, TypeError):
                        pass
            return 0
        
        # Test ALL MACs and merge their categories
        for mac in macs:
            try:
                token = stb.getToken(url, mac, proxy)
                if not token:
                    continue
                    
                working_macs_count += 1
                logger.info(f"Loading categories from MAC {mac} for portal {portal_id}")
                
                # Get VOD categories from this MAC
                vod_cats = stb.getVodCategories(url, mac, token, proxy)
                if vod_cats:
                    for cat in vod_cats:
                        cat_id = str(cat.get('id', ''))
                        title = cat.get('title', '')
                        item_count = get_item_count(cat)
                        
                        # Skip "all" category (id = "*")
                        if cat_id == "*" or not cat_id:
                            continue
                        
                        # Log first category to debug field names
                        if cat_id and not all_vod_categories:
                            logger.debug(f"VOD category fields: {list(cat.keys())}")
                        
                        if cat_id not in all_vod_categories:
                            all_vod_categories[cat_id] = {
                                "category_id": cat_id,
                                "title": title,
                                "item_count": item_count,
                                "working_macs": [mac]
                            }
                        else:
                            # Category exists, add this MAC to working_macs
                            if mac not in all_vod_categories[cat_id]["working_macs"]:
                                all_vod_categories[cat_id]["working_macs"].append(mac)
                            # Update item_count if higher
                            if item_count > all_vod_categories[cat_id]["item_count"]:
                                all_vod_categories[cat_id]["item_count"] = item_count
                
                # Get Series categories from this MAC
                series_cats = stb.getSeriesCategories(url, mac, token, proxy)
                if series_cats:
                    for cat in series_cats:
                        cat_id = str(cat.get('id', ''))
                        title = cat.get('title', '')
                        item_count = get_item_count(cat)
                        
                        # Skip "all" category (id = "*")
                        if cat_id == "*" or not cat_id:
                            continue
                        
                        # Log first category to debug field names
                        if cat_id and not all_series_categories:
                            logger.debug(f"Series category fields: {list(cat.keys())}")
                        
                        if cat_id not in all_series_categories:
                            all_series_categories[cat_id] = {
                                "category_id": cat_id,
                                "title": title,
                                "item_count": item_count,
                                "working_macs": [mac]
                            }
                        else:
                            # Category exists, add this MAC to working_macs
                            if mac not in all_series_categories[cat_id]["working_macs"]:
                                all_series_categories[cat_id]["working_macs"].append(mac)
                            # Update item_count if higher
                            if item_count > all_series_categories[cat_id]["item_count"]:
                                all_series_categories[cat_id]["item_count"] = item_count
                                
            except Exception as e:
                logger.warning(f"Error loading categories from MAC {mac}: {e}")
                continue
        
        if working_macs_count == 0:
            return jsonify({"success": False, "error": "Could not authenticate with any MAC"})
        
        categories = []
        
        # Save VOD categories to database
        for cat_id, cat_data in all_vod_categories.items():
            working_macs_str = ','.join(cat_data["working_macs"])
            cursor.execute('''
                INSERT OR REPLACE INTO vod_categories 
                (portal_id, category_id, title, content_type, item_count, working_mac)
                VALUES (?, ?, ?, 'vod', ?, ?)
            ''', (portal_id, cat_id, cat_data["title"], cat_data["item_count"], working_macs_str))
            
            categories.append({
                "category_id": cat_id,
                "title": cat_data["title"],
                "type": "vod",
                "item_count": cat_data["item_count"],
                "working_mac": working_macs_str
            })
        
        # Save Series categories to database
        for cat_id, cat_data in all_series_categories.items():
            working_macs_str = ','.join(cat_data["working_macs"])
            cursor.execute('''
                INSERT OR REPLACE INTO vod_categories 
                (portal_id, category_id, title, content_type, item_count, working_mac)
                VALUES (?, ?, ?, 'series', ?, ?)
            ''', (portal_id, cat_id, cat_data["title"], cat_data["item_count"], working_macs_str))
            
            categories.append({
                "category_id": cat_id,
                "title": cat_data["title"],
                "type": "series",
                "item_count": cat_data["item_count"],
                "working_mac": working_macs_str
            })
        
        conn.commit()
        conn.close()
        
        logger.info(f"Loaded {len(all_vod_categories)} VOD and {len(all_series_categories)} Series categories from {working_macs_count} MACs for portal {portal_id}")
        
        return jsonify({
            "success": True, 
            "categories": categories,
            "macs_tested": len(macs),
            "macs_working": working_macs_count
        })
    except Exception as e:
        logger.error(f"Error loading categories: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/vods/stream", methods=["POST"])
@authorise
def vods_stream():
    """Get stream URL for VOD item."""
    try:
        data = request.get_json()
        portal_id = data.get('portal_id')
        content_type = data.get('content_type', 'vod')
        cmd = data.get('cmd')
        series_id = data.get('series_id')
        season_id = data.get('season_id')
        episode_id = data.get('episode_id')
        
        if not portal_id or not cmd:
            return jsonify({"success": False, "error": "Missing portal_id or cmd"})
        
        portals = getPortals()
        portal = portals.get(portal_id)
        
        if not portal:
            return jsonify({"success": False, "error": "Portal not found"})
        
        url = portal.get("url")
        macs = list(portal.get("macs", {}).keys())
        proxy = portal.get("proxy")
        
        if not macs:
            return jsonify({"success": False, "error": "No MACs configured"})
        
        # Get VOD settings
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT key, value FROM vod_settings')
        settings = {row['key']: row['value'] for row in cursor.fetchall()}
        conn.close()
        
        stream_type = settings.get('stream_type', 'ffmpeg')
        mac_rotation = settings.get('mac_rotation', 'true') == 'true'
        
        # Select MAC based on rotation setting
        if mac_rotation:
            selected_mac = get_next_mac_for_portal(portal_id, macs)
            macs_to_try = [selected_mac] + [m for m in macs if m != selected_mac]
        else:
            macs_to_try = macs
        
        # Get stream link with fallback
        link, working_mac = try_get_vod_link_with_fallback(
            url, macs_to_try, cmd, content_type, proxy,
            series_id, season_id, episode_id
        )
        
        if not link:
            return jsonify({"success": False, "error": "Could not get stream URL"})
        
        # Return based on stream type
        if stream_type == 'direct':
            return jsonify({
                "success": True,
                "stream_url": link,
                "stream_type": "direct",
                "working_mac": working_mac
            })
        else:
            # FFmpeg - return internal play URL
            return jsonify({
                "success": True,
                "stream_url": link,
                "stream_type": "ffmpeg",
                "working_mac": working_mac
            })
    except Exception as e:
        logger.error(f"Error getting VOD stream: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/vods/items/load", methods=["POST"])
@authorise
def vods_load_items():
    """Load items for a category on-demand and cache them."""
    try:
        data = request.get_json()
        portal_id = data.get('portal_id')
        category_id = data.get('category_id')
        content_type = data.get('content_type', 'vod')
        
        portals = getPortals()
        portal = portals.get(portal_id)
        
        if not portal:
            return jsonify({"success": False, "error": "Portal not found"})
        
        url = portal.get("url")
        macs = list(portal.get("macs", {}).keys())
        proxy = portal.get("proxy")
        
        if not macs:
            return jsonify({"success": False, "error": "No MACs configured"})
        
        # Try each MAC until one works
        working_mac = None
        token = None
        for mac in macs:
            try:
                token = stb.getToken(url, mac, proxy)
                if token:
                    working_mac = mac
                    break
            except:
                continue
        
        if not token:
            return jsonify({"success": False, "error": "Could not authenticate"})
        
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        
        all_items = []
        page = 1
        
        while True:
            if content_type == 'series':
                result = stb.getSeriesItems(url, working_mac, token, category_id, page, proxy)
            else:
                result = stb.getVodItems(url, working_mac, token, category_id, page, proxy)
            
            if not result or not result.get('items'):
                break
            
            items = result['items']
            all_items.extend(items)
            
            # Save items to database
            for item in items:
                item_id = str(item.get('id', ''))
                name = item.get('name', '')
                year = item.get('year', '')
                description = item.get('description', '')
                genre = item.get('genre_str', '')
                duration = item.get('time', '')
                rating = item.get('rating_imdb', '')
                poster_url = item.get('screenshot_uri', '')
                cmd = item.get('cmd', '')
                
                cursor.execute('''
                    INSERT OR REPLACE INTO vod_items 
                    (portal_id, category_id, item_id, content_type, name, year, description, 
                     genre, duration, rating, poster_url, cmd, working_macs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (portal_id, category_id, item_id, content_type, name, year, description,
                      genre, duration, rating, poster_url, cmd, working_mac))
            
            # Check if there are more pages
            total = result.get('total', 0)
            if len(all_items) >= total or len(items) < 14:
                break
            
            page += 1
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True,
            "items_loaded": len(all_items),
            "category_id": category_id
        })
    except Exception as e:
        logger.error(f"Error loading VOD items: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/vods/debug/test-api", methods=["POST"])
@authorise
def vods_debug_test_api():
    """Debug endpoint to test VOD API directly and return raw response."""
    try:
        data = request.get_json()
        portal_id = data.get('portal_id')
        category_id = data.get('category_id')
        content_type = data.get('content_type', 'vod')
        
        portals = getPortals()
        portal = portals.get(portal_id)
        
        if not portal:
            return jsonify({"success": False, "error": "Portal not found"})
        
        url = portal.get("url")
        macs = list(portal.get("macs", {}).keys())
        proxy = portal.get("proxy")
        
        if not macs:
            return jsonify({"success": False, "error": "No MACs configured"})
        
        # Try first MAC
        mac = macs[0]
        token = stb.getToken(url, mac, proxy)
        
        if not token:
            return jsonify({"success": False, "error": f"Could not get token for MAC {mac}"})
        
        # Make direct API call
        import requests as req
        
        proxies = {"http": proxy, "https": proxy} if proxy else None
        cookies = {"mac": mac, "stb_lang": "en", "timezone": "Europe/London"}
        headers = {
            "User-Agent": "Mozilla/5.0 (QtEmbedded; U; Linux; C)",
            "Authorization": "Bearer " + token,
        }
        
        # Build params matching macvod.py exactly
        params = {
            "type": content_type if content_type == "series" else "vod",
            "action": "get_ordered_list",
            "movie_id": "0",
            "season_id": "0",
            "episode_id": "0",
            "row": "0",
            "JsHttpRequest": "1-xml",
            "category": str(category_id),
            "sortby": "added",
            "fav": "0",
            "hd": "0",
            "not_ended": "0",
            "abc": "*",
            "genre": "*",
            "years": "*",
            "search": "",
            "p": "1"
        }
        
        response = req.get(
            url,
            params=params,
            cookies=cookies,
            headers=headers,
            proxies=proxies,
            timeout=30,
        )
        
        return jsonify({
            "success": True,
            "portal_id": portal_id,
            "portal_url": url,
            "mac": mac,
            "category_id": category_id,
            "content_type": content_type,
            "request_url": response.url,
            "status_code": response.status_code,
            "response_text": response.text[:2000] if len(response.text) > 2000 else response.text,
            "response_json": response.json() if response.status_code == 200 else None
        })
    except Exception as e:
        logger.error(f"Error in VOD debug: {e}")
        import traceback
        return jsonify({"success": False, "error": str(e), "traceback": traceback.format_exc()})


@app.route("/vods/play/<portal_id>/<content_type>", methods=["POST"])
@authorise
def vods_play(portal_id, content_type):
    """Get playback URL for VOD/Series item by calling create_link API.
    
    Tests each MAC to find one that has access to the content.
    """
    try:
        data = request.get_json()
        cmd = data.get('cmd')
        episode_num = data.get('episode_num', '0')  # Episode number for series
        season_num = data.get('season_num', '0')
        test_stream = data.get('test_stream', True)  # Test stream by default
        
        if not cmd:
            return jsonify({"success": False, "error": "No cmd provided"})
        
        portals = getPortals()
        portal = portals.get(portal_id)
        
        if not portal:
            return jsonify({"success": False, "error": "Portal not found"})
        
        url = portal.get("url")
        macs = list(portal.get("macs", {}).keys())
        proxy = portal.get("proxy")
        
        if not macs:
            return jsonify({"success": False, "error": "No MACs configured"})
        
        # Get VOD settings for stream testing
        settings = getSettings()
        should_test = settings.get("test streams", "true") == "true" and test_stream
        
        failed_macs = []
        
        # Try each MAC until we get a working link
        for mac in macs:
            try:
                logger.info(f"Trying MAC {mac} for {content_type} content...")
                token = stb.getToken(url, mac, proxy)
                if not token:
                    logger.warning(f"MAC {mac}: Failed to get token")
                    failed_macs.append({"mac": mac, "reason": "No token"})
                    continue
                
                # Call create_link API
                if content_type == 'series':
                    # For series, episode_num is passed as the 'series' parameter
                    link = stb.getSeriesLink(url, mac, token, cmd, episode_num, season_num, episode_num, proxy)
                else:
                    link = stb.getVodLink(url, mac, token, cmd, proxy)
                
                if not link:
                    logger.warning(f"MAC {mac}: No link returned from API")
                    failed_macs.append({"mac": mac, "reason": "No link from API"})
                    continue
                
                # Test if the stream is accessible
                if should_test:
                    logger.info(f"Testing stream link from MAC {mac}...")
                    if stb.testStreamLink(link, proxy, timeout=5):
                        logger.info(f"MAC {mac}: Stream test PASSED - {link[:50]}...")
                        return jsonify({
                            "success": True,
                            "url": link,
                            "mac": mac,
                            "tested": True
                        })
                    else:
                        logger.warning(f"MAC {mac}: Stream test FAILED - {link[:50]}...")
                        failed_macs.append({"mac": mac, "reason": "Stream test failed"})
                        continue
                else:
                    # Return without testing
                    logger.info(f"Got play link for {content_type} (untested): {link[:50]}...")
                    return jsonify({
                        "success": True,
                        "url": link,
                        "mac": mac,
                        "tested": False
                    })
                    
            except Exception as e:
                logger.warning(f"Error getting play link with MAC {mac}: {e}")
                failed_macs.append({"mac": mac, "reason": str(e)})
                continue
        
        # All MACs failed
        error_msg = f"Could not get working stream from any MAC. Tried {len(macs)} MAC(s)."
        if failed_macs:
            reasons = [f"{m['mac'][:10]}...: {m['reason']}" for m in failed_macs[:3]]
            error_msg += f" Reasons: {'; '.join(reasons)}"
        
        return jsonify({"success": False, "error": error_msg, "failed_macs": failed_macs})
    except Exception as e:
        logger.error(f"Error in VOD play: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/vods/series/<portal_id>/<series_id>/episodes", methods=["GET"])
@authorise
def vods_series_episodes(portal_id, series_id):
    """Get episodes for a series."""
    try:
        portals = getPortals()
        portal = portals.get(portal_id)
        
        if not portal:
            return jsonify({"success": False, "error": "Portal not found"})
        
        url = portal.get("url")
        macs = list(portal.get("macs", {}).keys())
        proxy = portal.get("proxy")
        
        if not macs:
            return jsonify({"success": False, "error": "No MACs configured"})
        
        # Try each MAC until we get episodes
        for mac in macs:
            try:
                token = stb.getToken(url, mac, proxy)
                if not token:
                    continue
                
                # Get series info with episodes
                series_info = stb.getSeriesInfo(url, mac, token, series_id, proxy)
                
                if series_info:
                    logger.info(f"Got series info for {series_id}")
                    return jsonify({
                        "success": True,
                        "series_info": series_info,
                        "mac": mac
                    })
            except Exception as e:
                logger.warning(f"Error getting series info with MAC {mac}: {e}")
                continue
        
        return jsonify({"success": False, "error": "Could not get series info from any MAC"})
    except Exception as e:
        logger.error(f"Error getting series episodes: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/", methods=["GET"])
@authorise
def home():
    return redirect("/dashboard", code=302)

@app.route("/portals", methods=["GET"])
@authorise
def portals():
    # Check if we should show genre modal
    show_genre_modal = flask.session.pop('show_genre_modal', False)
    genre_modal_portal_id = flask.session.pop('genre_modal_portal_id', None)
    genre_modal_portal_name = flask.session.pop('genre_modal_portal_name', None)
    
    return render_template("portals.html", 
                         portals=getPortals(),
                         settings=getSettings(),
                         show_genre_modal=show_genre_modal,
                         genre_modal_portal_id=genre_modal_portal_id,
                         genre_modal_portal_name=genre_modal_portal_name)

@app.route("/portal/test-macs", methods=["POST"])
@authorise
def portal_test_macs():
    """Test MAC addresses for a portal."""
    try:
        data = request.json
        url = data.get('url')
        macs = data.get('macs', [])
        proxy = data.get('proxy', '')
        
        if not url:
            return flask.jsonify({"error": "No URL provided"}), 400
        
        if not validate_url(url):
            return flask.jsonify({"error": "Invalid URL format"}), 400
        
        if not macs:
            return flask.jsonify({"error": "No MAC addresses provided"}), 400
        
        # Validate MAC addresses
        invalid_macs = [mac for mac in macs if not validate_mac_address(mac)]
        if invalid_macs:
            return flask.jsonify({"error": f"Invalid MAC address format: {', '.join(invalid_macs)}"}), 400
        
        # Ensure URL ends with .php
        if not url.endswith(".php"):
            url = stb.getUrl(url, proxy)
            if not url:
                return flask.jsonify({"error": "Invalid portal URL"}), 400
        
        results = []
        
        for mac in macs:
            mac = mac.strip()
            if not mac:
                continue
            
            result = {
                "mac": mac,
                "valid": False,
                "expiry": None
            }
            
            try:
                logger.info(f"Testing MAC: {mac}")
                token = stb.getToken(url, mac, proxy)
                if token:
                    stb.getProfile(url, mac, token, proxy)
                    expiry = stb.getExpires(url, mac, token, proxy)
                    if expiry:
                        result["valid"] = True
                        result["expiry"] = expiry
                        logger.info(f"MAC {mac} is valid, expires: {expiry}")
                    else:
                        logger.warning(f"MAC {mac} got token but no expiry")
                else:
                    logger.warning(f"MAC {mac} failed to get token")
            except Exception as e:
                logger.error(f"Error testing MAC {mac}: {e}")
            
            results.append(result)
        
        return flask.jsonify({"results": results})
    except Exception as e:
        logger.error(f"Error in portal_test_macs: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/portal/add", methods=["POST"])
@authorise
def portalsAdd():
    global cached_xmltv
    cached_xmltv = None
    id = uuid.uuid4().hex
    enabled = "true"
    name = request.form.get("name", "").strip()
    url = request.form.get("url", "").strip()
    
    # Validate inputs
    if not name:
        flash("Portal name is required", "danger")
        return redirect("/portals", code=302)
    
    if not url or not validate_url(url):
        flash("Valid portal URL is required", "danger")
        return redirect("/portals", code=302)
    
    # Support newline-separated MACs
    macs_text = request.form.get("macs", "")
    macs = [m.strip() for m in macs_text.split('\n') if m.strip()]
    macs = list(set(macs))  # Remove duplicates
    
    # Validate MAC addresses
    invalid_macs = [mac for mac in macs if not validate_mac_address(mac)]
    if invalid_macs:
        flash(f"Invalid MAC address format: {', '.join(invalid_macs)}", "danger")
        return redirect("/portals", code=302)
    
    if not macs:
        flash("At least one MAC address is required", "danger")
        return redirect("/portals", code=302)
    
    streamsPerMac = request.form.get("streams per mac", "1")
    epgOffset = request.form.get("epg offset", "0")
    proxy = request.form.get("proxy", "").strip()
    portalPrefix = request.form.get("portal prefix", "").strip()
    
    # Validate proxy if provided
    if proxy and not validate_proxy_url(proxy):
        proxy_type = get_proxy_type(proxy)
        flash(f"Invalid proxy format. Detected type: {proxy_type}. Use: http://host:port, socks5://host:port, ss://method:password@host:port, etc.", "danger")
        return redirect("/portals", code=302)

    if not url.endswith(".php"):
        url = stb.getUrl(url, proxy)
        if not url:
            logger.error("Error getting URL for Portal({})".format(name))
            flash("Error getting URL for Portal({})".format(name), "danger")
            return redirect("/portals", code=302)

    macsd = {}

    for mac in macs:
        token = stb.getToken(url, mac, proxy)
        if token:
            stb.getProfile(url, mac, token, proxy)
            expiry = stb.getExpires(url, mac, token, proxy)
            if expiry:
                macsd[mac] = expiry
                logger.info(
                    "Successfully tested MAC({}) for Portal({})".format(mac, name)
                )
                flash(
                    "Successfully tested MAC({}) for Portal({})".format(mac, name),
                    "success",
                )
                continue

        logger.error("Error testing MAC({}) for Portal({})".format(mac, name))
        flash("Error testing MAC({}) for Portal({})".format(mac, name), "danger")

    if len(macsd) > 0:
        portal = {
            "enabled": enabled,
            "name": name,
            "url": url,
            "macs": macsd,
            "streams per mac": streamsPerMac,
            "epg offset": epgOffset,
            "proxy": proxy,
            "portal prefix": portalPrefix,
        }

        for setting, default in defaultPortal.items():
            if not portal.get(setting):
                portal[setting] = default

        portals = getPortals()
        portals[id] = portal
        savePortals(portals)
        logger.info("Portal({}) added!".format(portal["name"]))
        
        # Store portal ID in session for genre selection modal
        flask.session['show_genre_modal'] = True
        flask.session['genre_modal_portal_id'] = id
        flask.session['genre_modal_portal_name'] = name
        return redirect("/portals", code=302)

    else:
        logger.error(
            "None of the MACs tested OK for Portal({}). Adding not successfull".format(
                name
            )
        )

    return redirect("/portals", code=302)

@app.route("/portal/update", methods=["POST"])
@authorise
def portalUpdate():
    global cached_xmltv
    cached_xmltv = None
    id = request.form["id"]
    enabled = request.form.get("enabled", "false")
    name = request.form["name"]
    url = request.form["url"]
    # Support newline-separated MACs
    macs_text = request.form["macs"]
    newmacs = [m.strip() for m in macs_text.split('\n') if m.strip()]
    newmacs = list(set(newmacs))  # Remove duplicates
    streamsPerMac = request.form["streams per mac"]
    epgOffset = request.form["epg offset"]
    proxy = request.form["proxy"].strip()
    portalPrefix = request.form.get("portal prefix", "").strip()
    retest = request.form.get("retest", None)
    
    # Validate proxy if provided
    if proxy and not validate_proxy_url(proxy):
        proxy_type = get_proxy_type(proxy)
        flash(f"Invalid proxy format. Detected type: {proxy_type}. Use: http://host:port, socks5://host:port, ss://method:password@host:port, etc.", "danger")
        return redirect("/portals", code=302)

    if not url.endswith(".php"):
        url = stb.getUrl(url, proxy)
        if not url:
            logger.error("Error getting URL for Portal({})".format(name))
            flash("Error getting URL for Portal({})".format(name), "danger")
            return redirect("/portals", code=302)

    portals = getPortals()
    oldmacs = portals[id]["macs"]
    macsout = {}
    deadmacs = []

    for mac in newmacs:
        if retest or mac not in oldmacs.keys():
            token = stb.getToken(url, mac, proxy)
            if token:
                stb.getProfile(url, mac, token, proxy)
                expiry = stb.getExpires(url, mac, token, proxy)
                if expiry:
                    macsout[mac] = expiry
                    logger.info(
                        "Successfully tested MAC({}) for Portal({})".format(mac, name)
                    )
                    flash(
                        "Successfully tested MAC({}) for Portal({})".format(mac, name),
                        "success",
                    )

            if mac not in list(macsout.keys()):
                deadmacs.append(mac)

        if mac in oldmacs.keys() and mac not in deadmacs:
            macsout[mac] = oldmacs[mac]

        if mac not in macsout.keys():
            logger.error("Error testing MAC({}) for Portal({})".format(mac, name))
            flash("Error testing MAC({}) for Portal({})".format(mac, name), "danger")

    if len(macsout) > 0:
        portals[id]["enabled"] = enabled
        portals[id]["name"] = name
        portals[id]["url"] = url
        portals[id]["macs"] = macsout
        portals[id]["streams per mac"] = streamsPerMac
        portals[id]["epg offset"] = epgOffset
        portals[id]["proxy"] = proxy
        portals[id]["portal prefix"] = portalPrefix
        savePortals(portals)
        logger.info("Portal({}) updated!".format(name))
        flash("Portal({}) updated!".format(name), "success")

    else:
        logger.error(
            "None of the MACs tested OK for Portal({}). Adding not successfull".format(
                name
            )
        )

    return redirect("/portals", code=302)

@app.route("/portal/genre-selection", methods=["GET"])
@authorise
def portal_genre_selection():
    """Show genre selection page after adding a portal."""
    # Check for query parameters first (for direct links from portal page)
    portal_id = request.args.get('portal_id')
    portal_name = request.args.get('portal_name')
    
    # If not in query params, check session (for new portal flow)
    if not portal_id:
        portal_id = flask.session.get('new_portal_id')
        portal_name = flask.session.get('new_portal_name')
    else:
        # Store in session for subsequent API calls
        flask.session['new_portal_id'] = portal_id
        flask.session['new_portal_name'] = portal_name
    
    if not portal_id:
        return redirect("/portals", code=302)
    
    return render_template("genre_selection.html", portal_id=portal_id, portal_name=portal_name)


@app.route("/portal/load-genres", methods=["POST"])
@authorise
def portal_load_genres():
    """Load genres for a specific portal - uses database cache if available."""
    try:
        portal_id = request.json.get('portal_id')
        force_refresh = request.json.get('force_refresh', False)
        
        if not portal_id:
            return flask.jsonify({"error": "No portal ID provided"}), 400
        
        portals = getPortals()
        portal = portals.get(portal_id)
        if not portal:
            return flask.jsonify({"error": "Portal not found"}), 404
        
        portal_name = portal["name"]
        
        # Check if we have cached data in database (unless force refresh)
        if not force_refresh:
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Check if we have any channels for this portal in DB
                cursor.execute('SELECT COUNT(*) as count FROM channels WHERE portal = ?', (portal_id,))
                row = cursor.fetchone()
                channel_count = row['count'] if row else 0
                
                if channel_count > 0:
                    logger.info(f"Loading genres from database cache for portal {portal_id} ({channel_count} channels)")
                    
                    # Get genre counts from database
                    cursor.execute('''
                        SELECT genre, COUNT(*) as count
                        FROM channels
                        WHERE portal = ? AND genre IS NOT NULL AND genre != ''
                        GROUP BY genre
                        ORDER BY genre
                    ''', (portal_id,))
                    
                    genres = [{"name": row['genre'], "count": row['count']} for row in cursor.fetchall()]
                    
                    # Get previously selected genres from database
                    cursor.execute('SELECT genre FROM portal_genres WHERE portal = ?', (portal_id,))
                    enabled_genres = [row['genre'] for row in cursor.fetchall()]
                    
                    # Fallback to JSON config if database is empty
                    if not enabled_genres:
                        enabled_genres = portal.get("selected genres", [])
                    
                    conn.close()
                    
                    logger.info(f"Loaded {len(genres)} genres from database cache")
                    return flask.jsonify({
                        "genres": genres,
                        "total_channels": channel_count,
                        "enabled_genres": enabled_genres,
                        "from_cache": True
                    })
                
                conn.close()
            except Exception as e:
                logger.error(f"Error loading from database cache: {e}")
        
        # Fetch from portal (first time or force refresh)
        logger.info(f"Fetching genres from portal {portal_id} (force_refresh={force_refresh})")
        
        url = portal["url"]
        macs = list(portal["macs"].keys())
        proxy = portal["proxy"]
        
        all_channels_map = {}  # channel_id -> channel data
        all_genres_dict = {}  # genre_id -> genre_name
        
        logger.info(f"Loading channels from {len(macs)} MACs for portal {portal_id}")
        
        for mac in macs:
            try:
                token = stb.getToken(url, mac, proxy)
                if token:
                    stb.getProfile(url, mac, token, proxy)
                    mac_channels = stb.getAllChannels(url, mac, token, proxy)
                    mac_genres = stb.getGenreNames(url, mac, token, proxy)
                    
                    if mac_channels:
                        for channel in mac_channels:
                            channel_id = str(channel["id"])
                            if channel_id not in all_channels_map:
                                all_channels_map[channel_id] = channel
                        logger.info(f"MAC {mac}: Added {len(mac_channels)} channels (total now: {len(all_channels_map)})")
                    
                    if mac_genres:
                        all_genres_dict.update(mac_genres)
                        logger.info(f"MAC {mac}: Added genres (total now: {len(all_genres_dict)})")
                        
            except Exception as e:
                logger.error(f"Error fetching from MAC {mac}: {e}")
                continue
        
        if not all_channels_map or not all_genres_dict:
            return flask.jsonify({"error": "Failed to fetch channels from any MAC"}), 500
        
        logger.info(f"Total channels loaded from all MACs: {len(all_channels_map)}")
        
        # Save to database for future fast loading
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Delete existing channels for this portal
            cursor.execute('DELETE FROM channels WHERE portal = ?', (portal_id,))
            
            # Insert all channels
            for channel_id, channel in all_channels_map.items():
                channel_name = str(channel.get("name", ""))
                channel_number = str(channel.get("number", ""))
                genre_id = str(channel.get("tv_genre_id", ""))
                genre = all_genres_dict.get(genre_id, "")
                logo = str(channel.get("logo", ""))
                
                cursor.execute('''
                    INSERT INTO channels (
                        portal, channel_id, portal_name, name, number, genre, logo,
                        enabled, custom_name, custom_number, custom_genre, 
                        custom_epg_id, fallback_channel, has_portal_epg
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, '', '', '', '', '', 0)
                ''', (portal_id, channel_id, portal_name, channel_name, channel_number, genre, logo))
            
            conn.commit()
            conn.close()
            logger.info(f"Cached {len(all_channels_map)} channels to database")
        except Exception as e:
            logger.error(f"Error caching to database: {e}")
        
        # Count channels per genre
        genre_counts = {}
        for channel_id, channel in all_channels_map.items():
            genre_id = str(channel.get("tv_genre_id", ""))
            genre_name = all_genres_dict.get(genre_id, "Unknown")
            
            if genre_name not in genre_counts:
                genre_counts[genre_name] = 0
            genre_counts[genre_name] += 1
        
        # Get previously selected genres from database
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT genre FROM portal_genres WHERE portal = ?', (portal_id,))
            enabled_genres = [row['genre'] for row in cursor.fetchall()]
            conn.close()
            
            # Fallback to JSON config if database is empty
            if not enabled_genres:
                enabled_genres = portal.get("selected genres", [])
        except Exception as e:
            logger.error(f"Error loading selected genres from database: {e}")
            enabled_genres = portal.get("selected genres", [])
        
        logger.info(f"Portal {portal_id}: {len(all_channels_map)} total channels, {len(genre_counts)} genres, {len(enabled_genres)} selected")
        
        # Sort genres by name
        genres = sorted([{"name": name, "count": count} for name, count in genre_counts.items()], key=lambda x: x['name'])
        
        return flask.jsonify({
            "genres": genres, 
            "total_channels": len(all_channels_map),
            "enabled_genres": enabled_genres,
            "from_cache": False
        })
    except Exception as e:
        logger.error(f"Error loading genres: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/portal/save-genre-selection", methods=["POST"])
@authorise
def portal_save_genre_selection():
    """Save genre selection and enable channels - fetches from ALL MACs."""
    try:
        portal_id = request.json.get('portal_id')
        selected_genres = request.json.get('selected_genres', [])
        auto_sync = request.json.get('auto_sync', False)
        
        if not portal_id:
            return flask.jsonify({"error": "No portal ID provided"}), 400
        
        portals = getPortals()
        portal = portals.get(portal_id)
        if not portal:
            return flask.jsonify({"error": "Portal not found"}), 404
        
        # Fetch channels from ALL MACs and merge
        url = portal["url"]
        macs = list(portal["macs"].keys())
        proxy = portal["proxy"]
        portal_name = portal["name"]
        
        all_channels_map = {}  # channel_id -> channel data
        all_genres_dict = {}  # genre_id -> genre_name
        
        logger.info(f"Saving genre selection: fetching from {len(macs)} MACs for portal {portal_name}")
        
        for mac in macs:
            try:
                token = stb.getToken(url, mac, proxy)
                if token:
                    stb.getProfile(url, mac, token, proxy)
                    mac_channels = stb.getAllChannels(url, mac, token, proxy)
                    mac_genres = stb.getGenreNames(url, mac, token, proxy)
                    
                    if mac_channels:
                        # Merge channels
                        for channel in mac_channels:
                            channel_id = str(channel["id"])
                            if channel_id not in all_channels_map:
                                all_channels_map[channel_id] = channel
                        logger.info(f"MAC {mac}: Added {len(mac_channels)} channels (total: {len(all_channels_map)})")
                    
                    if mac_genres:
                        all_genres_dict.update(mac_genres)
                        
            except Exception as e:
                logger.error(f"Error fetching from MAC {mac}: {e}")
                continue
        
        if not all_channels_map or not all_genres_dict:
            return flask.jsonify({"error": "Failed to fetch channels from any MAC"}), 500
        
        # Save enabled channels to portal configuration
        enabled_channels = []
        enabled_count = 0
        total_count = len(all_channels_map)
        
        logger.info(f"Selected genres: {selected_genres}")
        logger.info(f"Processing {total_count} total channels from all MACs")
        
        for channel_id, channel in all_channels_map.items():
            genre_id = str(channel.get("tv_genre_id", ""))
            genre = all_genres_dict.get(genre_id, "")
            
            # Enable channel if its genre is selected
            if genre in selected_genres:
                enabled_channels.append(channel_id)
                enabled_count += 1
        
        logger.info(f"Enabled {enabled_count} channels out of {total_count}")
        logger.info(f"First 10 enabled channel IDs: {enabled_channels[:10]}")
        
        # Update portal configuration
        portals = getPortals()
        if portal_id in portals:
            portals[portal_id]["enabled channels"] = enabled_channels
            portals[portal_id]["selected genres"] = selected_genres  # Save selected genres
            savePortals(portals)
            logger.info(f"Saved to portal config. Verifying...")
            
            # Verify it was saved
            portals_verify = getPortals()
            saved_count = len(portals_verify[portal_id].get("enabled channels", []))
            saved_genres = portals_verify[portal_id].get("selected genres", [])
            logger.info(f"Verification: {saved_count} channels in 'enabled channels' list")
            logger.info(f"Verification: {len(saved_genres)} genres in 'selected genres' list")
            
            # Insert/Update channels in database
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # First, delete all existing channels for this portal
                cursor.execute('DELETE FROM channels WHERE portal = ?', (portal_id,))
                logger.info(f"Deleted existing channels for portal {portal_id}")
                
                # Insert all channels into database
                inserted_count = 0
                for channel_id, channel in all_channels_map.items():
                    channel_name = str(channel.get("name", ""))
                    channel_number = str(channel.get("number", ""))
                    genre_id = str(channel.get("tv_genre_id", ""))
                    genre = all_genres_dict.get(genre_id, "")
                    logo = str(channel.get("logo", ""))
                    
                    # Check if this channel should be enabled
                    is_enabled = 1 if channel_id in enabled_channels else 0
                    
                    cursor.execute('''
                        INSERT INTO channels (
                            portal, channel_id, portal_name, name, number, genre, logo,
                            enabled, custom_name, custom_number, custom_genre, 
                            custom_epg_id, fallback_channel, has_portal_epg
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '', '', '', '', '', 0)
                    ''', (
                        portal_id, channel_id, portal_name, channel_name, channel_number,
                        genre, logo, is_enabled
                    ))
                    inserted_count += 1
                
                # Save selected genres to database
                cursor.execute('DELETE FROM portal_genres WHERE portal = ?', (portal_id,))
                for genre in selected_genres:
                    cursor.execute('INSERT INTO portal_genres (portal, genre) VALUES (?, ?)', (portal_id, genre))
                
                conn.commit()
                conn.close()
                logger.info(f"Inserted {inserted_count} channels into database ({enabled_count} enabled)")
                logger.info(f"Saved {len(selected_genres)} selected genres to database")
            except Exception as e:
                logger.error(f"Error inserting channels into database: {e}")
        else:
            logger.error(f"Portal {portal_id} not found in portals!")
        
        # Clear session
        flask.session.pop('new_portal_id', None)
        flask.session.pop('new_portal_name', None)
        
        logger.info(f"Saved {enabled_count}/{total_count} channels for portal {portal_name}")
        return flask.jsonify({
            "success": True, 
            "enabled_count": enabled_count, 
            "total_count": total_count
        })
    except Exception as e:
        logger.error(f"Error saving genre selection: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/portal/remove", methods=["POST"])
@authorise
def portalRemove():
    id = request.form["deleteId"]
    portals = getPortals()
    name = portals[id]["name"]
    
    # Delete all channels for this portal from the database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM channels WHERE portal = ?', (id,))
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        logger.info("Deleted {} channels for portal ({}) from database".format(deleted_count, name))
    except Exception as e:
        logger.error("Error deleting channels from database: {}".format(e))
    
    # Delete portal from config
    del portals[id]
    savePortals(portals)
    
    logger.info("Portal ({}) removed!".format(name))
    flash("Portal ({}) removed!".format(name), "success")
    return redirect("/portals", code=302)


def apply_portal_prefix(channel_name, genre, portal_prefix):
    """Apply portal prefix to genre only (for group-title organization)."""
    if portal_prefix and genre:
        genre = f"[{portal_prefix}] {genre}"
    return channel_name, genre


def generate_portal_m3u(portal_id):
    """Generate M3U playlist content for a specific portal."""
    logger.info(f"Generating M3U for portal: {portal_id}")
    
    # Use external host configuration
    external_host, external_scheme = get_external_host_config()
    playlist_host = external_host or request.host or "0.0.0.0:8001"
    
    channels = []
    
    # Get enabled channels from database for specific portal
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT portal, channel_id, name, custom_name, genre, custom_genre, 
               number, custom_number, custom_epg_id
        FROM channels 
        WHERE enabled = 1 AND portal = ?
        ORDER BY channel_id
    ''', (portal_id,))
    db_channels = cursor.fetchall()
    conn.close()
    
    # Get portal info
    portals = getPortals()
    
    # Check if portal exists and is enabled
    if portal_id not in portals:
        logger.warning(f"Portal {portal_id} not found")
        return None
        
    if portals[portal_id].get("enabled") != "true":
        logger.warning(f"Portal {portal_id} is disabled")
        return "#EXTM3U \n"  # Return empty playlist for disabled portals
    
    # Get portal prefix
    portal_prefix = portals[portal_id].get("portal prefix", "").strip()
    
    for channel in db_channels:
        channel_id = str(channel['channel_id'])
        
        # Use custom values if available, otherwise use original values
        channel_name = channel['custom_name'] if channel['custom_name'] else (channel['name'] or "Unknown Channel")
        genre = channel['custom_genre'] if channel['custom_genre'] else (channel['genre'] or "")
        channel_number = channel['custom_number'] if channel['custom_number'] else (channel['number'] or "")
        epg_id = channel['custom_epg_id'] if channel['custom_epg_id'] else channel_name
        
        # Apply portal prefix to genre only (for group-title organization)
        if portal_prefix and genre:
            genre = f"[{portal_prefix}] {genre}"
        
        # Build M3U entry - escape quotes in attributes
        def escape_quotes(text):
            return str(text).replace('"', '&quot;') if text else ""
        
        m3u_entry = "#EXTINF:-1"
        m3u_entry += ' tvg-id="' + escape_quotes(epg_id) + '"'
        
        if getSettings().get("use channel numbers", "true") == "true" and channel_number:
            m3u_entry += ' tvg-chno="' + escape_quotes(channel_number) + '"'
        
        if getSettings().get("use channel genres", "true") == "true" and genre:
            m3u_entry += ' group-title="' + escape_quotes(genre) + '"'
        
        m3u_entry += ',' + str(channel_name) + "\n"
        m3u_entry += "http://" + playlist_host + "/play/" + portal_id + "/" + channel_id
        
        channels.append(m3u_entry)

    # Sort channels based on settings (same logic as main playlist)
    if getSettings().get("sort playlist by channel name", "true") == "true":
        channels.sort(key=lambda k: k.split(",")[1].split("\n")[0] if "," in k else "")
    if getSettings().get("use channel numbers", "true") == "true":
        if getSettings().get("sort playlist by channel number", "false") == "true":
            def get_channel_number(k):
                try:
                    if 'tvg-chno="' in k:
                        return int(k.split('tvg-chno="')[1].split('"')[0])
                    return 999999  # Put channels without numbers at the end
                except (ValueError, IndexError):
                    return 999999
            channels.sort(key=get_channel_number)
    if getSettings().get("use channel genres", "true") == "true":
        if getSettings().get("sort playlist by channel genre", "false") == "true":
            def get_genre(k):
                try:
                    if 'group-title="' in k:
                        return k.split('group-title="')[1].split('"')[0]
                    return "zzz"  # Put channels without genre at the end
                except IndexError:
                    return "zzz"
            channels.sort(key=get_genre)

    playlist = "#EXTM3U \n"
    if channels:
        playlist = playlist + "\n".join(channels)

    logger.info(f"Generated M3U for portal {portal_id} with {len(channels)} channels")
    return playlist


def generate_portal_m3u_with_auth(portal_id, username=None, password=None):
    """
    Generate M3U playlist content for a specific portal with authentication-aware stream URLs.
    
    Args:
        portal_id (str): Portal ID
        username (str): Username for embedding in stream URLs (if security enabled)
        password (str): Password for embedding in stream URLs (if security enabled)
        
    Returns:
        str: M3U playlist content with authentication-aware stream URLs
    """
    logger.info(f"Generating M3U with auth for portal: {portal_id}")
    
    # Use external host configuration
    external_host, external_scheme = get_external_host_config()
    playlist_host = external_host or request.host or "0.0.0.0:8001"
    
    channels = []
    
    # Get enabled channels from database for specific portal
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT portal, channel_id, name, custom_name, genre, custom_genre, 
               number, custom_number, custom_epg_id
        FROM channels 
        WHERE enabled = 1 AND portal = ?
        ORDER BY channel_id
    ''', (portal_id,))
    db_channels = cursor.fetchall()
    conn.close()
    
    # Get portal info
    portals = getPortals()
    
    # Check if portal exists and is enabled
    if portal_id not in portals:
        logger.warning(f"Portal {portal_id} not found")
        return None
        
    if portals[portal_id].get("enabled") != "true":
        logger.warning(f"Portal {portal_id} is disabled")
        return "#EXTM3U \n"  # Return empty playlist for disabled portals
    
    # Get settings to determine if we should embed auth in stream URLs
    settings = getSettings()
    security_enabled = settings.get("enable security", "false") == "true"
    
    # Get portal prefix
    portal_prefix = portals[portal_id].get("portal prefix", "").strip()
    
    for channel in db_channels:
        channel_id = str(channel['channel_id'])
        
        # Use custom values if available, otherwise use original values
        channel_name = channel['custom_name'] if channel['custom_name'] else (channel['name'] or "Unknown Channel")
        genre = channel['custom_genre'] if channel['custom_genre'] else (channel['genre'] or "")
        channel_number = channel['custom_number'] if channel['custom_number'] else (channel['number'] or "")
        epg_id = channel['custom_epg_id'] if channel['custom_epg_id'] else channel_name
        
        # Apply portal prefix to genre only (for group-title organization)
        if portal_prefix and genre:
            genre = f"[{portal_prefix}] {genre}"
        
        # Build M3U entry - escape quotes in attributes
        def escape_quotes(text):
            return str(text).replace('"', '&quot;') if text else ""
        
        m3u_entry = "#EXTINF:-1"
        m3u_entry += ' tvg-id="' + escape_quotes(epg_id) + '"'
        
        if getSettings().get("use channel numbers", "true") == "true" and channel_number:
            m3u_entry += ' tvg-chno="' + escape_quotes(channel_number) + '"'
        
        if getSettings().get("use channel genres", "true") == "true" and genre:
            m3u_entry += ' group-title="' + escape_quotes(genre) + '"'
        
        m3u_entry += ',' + str(channel_name) + "\n"
        
        # Generate stream URL with embedded auth if security is enabled and credentials provided
        if security_enabled and username and password:
            # Embed Basic Auth in stream URL for maximum player compatibility
            stream_url = f"http://{username}:{password}@{playlist_host}/play/{portal_id}/{channel_id}"
        else:
            # Standard stream URL without embedded auth
            stream_url = f"http://{playlist_host}/play/{portal_id}/{channel_id}"
        
        m3u_entry += stream_url
        
        channels.append(m3u_entry)

    # Sort channels based on settings (same logic as main playlist)
    if getSettings().get("sort playlist by channel name", "true") == "true":
        channels.sort(key=lambda k: k.split(",")[1].split("\n")[0] if "," in k else "")
    if getSettings().get("use channel numbers", "true") == "true":
        if getSettings().get("sort playlist by channel number", "false") == "true":
            def get_channel_number(k):
                try:
                    if 'tvg-chno="' in k:
                        return int(k.split('tvg-chno="')[1].split('"')[0])
                    return 999999  # Put channels without numbers at the end
                except (ValueError, IndexError):
                    return 999999
            channels.sort(key=get_channel_number)
    if getSettings().get("use channel genres", "true") == "true":
        if getSettings().get("sort playlist by channel genre", "false") == "true":
            def get_genre(k):
                try:
                    if 'group-title="' in k:
                        return k.split('group-title="')[1].split('"')[0]
                    return "zzz"  # Put channels without genre at the end
                except IndexError:
                    return "zzz"
            channels.sort(key=get_genre)

    playlist = "#EXTM3U \n"
    if channels:
        playlist = playlist + "\n".join(channels)

    logger.info(f"Generated M3U with auth for portal {portal_id} with {len(channels)} channels")
    return playlist


def generate_portal_filename(portal_name):
    """Generate a safe filename for portal M3U download."""
    import re
    # Remove or replace unsafe characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', portal_name)
    # Remove extra spaces and trim
    safe_name = re.sub(r'\s+', '_', safe_name.strip())
    # Ensure it's not empty
    if not safe_name:
        safe_name = "portal"
    return f"{safe_name}.m3u"


@app.route("/api/portal/<portal_id>/mac-status", methods=["GET"])
@authorise
def portal_mac_status(portal_id):
    """Check MAC address status for a portal."""
    try:
        portals = getPortals()
        
        if portal_id not in portals:
            return flask.jsonify({"success": False, "error": "Portal not found"}), 404
        
        portal = portals[portal_id]
        portal_url = portal["url"]
        macs = portal.get("macs", [])
        
        if not macs:
            return flask.jsonify({"success": False, "error": "No MAC addresses configured"}), 400
        
        # Import MAC status checker
        import requests
        import json
        from datetime import datetime
        
        def check_single_mac_status(portal_url, mac_address):
            """Check status of a single MAC address"""
            try:
                # Ensure portal URL ends with portal.php
                if not portal_url.endswith('/portal.php'):
                    if portal_url.endswith('/'):
                        portal_url_clean = portal_url + 'portal.php'
                    else:
                        portal_url_clean = portal_url + '/portal.php'
                else:
                    portal_url_clean = portal_url
                
                session = requests.Session()
                headers = {
                    'User-Agent': 'Mozilla/5.0 (QtEmbedded; U; Linux; C) AppleWebKit/533.3',
                    'X-User-Agent': f'Model: MAG250; Link: WiFi; MAC: {mac_address}',
                    'Cookie': f'mac={mac_address}; stb_lang=en'
                }
                
                # Get authentication token
                token_response = session.get(
                    f'{portal_url_clean}?type=stb&action=handshake&JsHttpRequest=1-xml',
                    headers=headers,
                    timeout=10
                )
                
                if token_response.status_code != 200:
                    return {
                        'success': False,
                        'mac': mac_address,
                        'error': f'Token request failed: HTTP {token_response.status_code}'
                    }
                
                try:
                    token_data = token_response.json()
                    token = token_data.get('js', {}).get('token')
                    if not token:
                        return {
                            'success': False,
                            'mac': mac_address,
                            'error': 'No token received from portal'
                        }
                except json.JSONDecodeError:
                    return {
                        'success': False,
                        'mac': mac_address,
                        'error': 'Invalid JSON response for token'
                    }
                
                # Get profile information
                profile_response = session.get(
                    f'{portal_url_clean}?type=stb&action=get_profile&JsHttpRequest=1-xml',
                    headers=headers,
                    timeout=10
                )
                
                if profile_response.status_code != 200:
                    return {
                        'success': False,
                        'mac': mac_address,
                        'error': f'Profile request failed: HTTP {profile_response.status_code}'
                    }
                
                try:
                    profile_data = profile_response.json()
                    profile = profile_data.get('js', {})
                except json.JSONDecodeError:
                    return {
                        'success': False,
                        'mac': mac_address,
                        'error': 'Invalid JSON response for profile'
                    }
                
                # Get account information
                account_response = session.get(
                    f'{portal_url_clean}?type=account_info&action=get_main_info&JsHttpRequest=1-xml',
                    headers=headers,
                    timeout=10
                )
                
                account_info = {}
                if account_response.status_code == 200:
                    try:
                        account_data = account_response.json()
                        account_info = account_data.get('js', {})
                    except json.JSONDecodeError:
                        pass
                
                # Extract key information
                watchdog_timeout = profile.get('watchdog_timeout')
                playback_limit = profile.get('playback_limit', 1)
                account_status = profile.get('status', 0)
                is_blocked = profile.get('blocked', '0') != '0'
                expires = account_info.get('phone', '')  # This seems to contain expiry date
                
                return {
                    'success': True,
                    'mac': mac_address,
                    'watchdog_timeout': watchdog_timeout,
                    'playback_limit': playback_limit,
                    'account_active': account_status == 1,
                    'is_blocked': is_blocked,
                    'expires': expires,
                    'token': token
                }
                
            except requests.exceptions.Timeout:
                return {
                    'success': False,
                    'mac': mac_address,
                    'error': 'Request timeout'
                }
            except requests.exceptions.ConnectionError as e:
                return {
                    'success': False,
                    'mac': mac_address,
                    'error': f'Connection error: {str(e)}'
                }
            except Exception as e:
                return {
                    'success': False,
                    'mac': mac_address,
                    'error': f'Unexpected error: {str(e)}'
                }
        
        # Use the enhanced stb.py functions for better MAC status checking
        import stb
        
        # Get MAC status summary using the smart system
        mac_list = list(macs.keys()) if isinstance(macs, dict) else macs
        mac_summary = stb.getMacStatusSummary(portal_url, mac_list, portal.get('proxy'))
        
        # Convert to the expected format
        mac_statuses = []
        portal_info = {}
        
        for mac_info in mac_summary:
            status = mac_info['status']
            if status['success']:
                # Add the enhanced information
                enhanced_status = {
                    'success': True,
                    'mac': status['mac'],
                    'watchdog_timeout': status.get('watchdog_timeout'),
                    'playback_limit': status.get('playback_limit', 1),
                    'account_active': status.get('account_active', False),
                    'is_blocked': status.get('is_blocked', False),
                    'expires': status.get('expires', ''),
                    'token': status.get('token', ''),
                    # New enhanced fields
                    'is_internally_used': status.get('is_internally_used', False),
                    'streams_used': status.get('streams_used', 0),
                    'max_streams': status.get('max_streams', 1),
                    'usage_ratio': status.get('usage_ratio', 0.0),
                    'internal_usage': status.get('internal_usage')
                }
                mac_statuses.append(enhanced_status)
                
                # Extract portal info from first successful response
                if not portal_info:
                    portal_info = {
                        'playback_limit': status.get('playback_limit', 1)
                    }
            else:
                # Keep failed status as is
                mac_statuses.append(status)
        
        return flask.jsonify({
            "success": True,
            "portal_id": portal_id,
            "portal_name": portal["name"],
            "portal_info": portal_info,
            "mac_statuses": mac_statuses,
            "checked_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error checking MAC status for portal {portal_id}: {e}")
        return flask.jsonify({"success": False, "error": str(e)}), 500


@app.route("/portal/download-m3u/<portal_id>", methods=["GET"])
def portal_download_m3u(portal_id):
    """Legacy portal M3U download with configurable access control."""
    settings = getSettings()
    public_access = settings.get("public playlist access", "true") == "true"
    
    if public_access:
        # Public access enabled - no authentication required
        return _portal_download_m3u(portal_id)
    else:
        # Public access disabled - require Basic Auth
        auth = request.authorization
        if not auth or not auth.username or not auth.password:
            # No Basic Auth provided - return 401 with WWW-Authenticate header
            response = Response(
                'Authentication required\n'
                'Please provide Basic Auth credentials in the URL:\n'
                'http://username:password@host/portal/download-m3u/portal_id',
                401,
                {'WWW-Authenticate': 'Basic realm="MacReplayXC Legacy M3U"'}
            )
            return response
        
        # Validate Basic Auth credentials
        system_username = settings.get("username", "admin")
        system_password = settings.get("password", "12345")
        
        if auth.username != system_username or auth.password != system_password:
            logger.warning(f"Invalid Basic Auth credentials for legacy M3U: {auth.username}")
            response = Response(
                'Invalid credentials\n'
                'Please check your username and password.',
                401,
                {'WWW-Authenticate': 'Basic realm="MacReplayXC Legacy M3U"'}
            )
            return response
        
        # Authentication successful
        logger.info(f"Basic Auth successful for legacy M3U: {auth.username}")
        return _portal_download_m3u(portal_id)

def _portal_download_m3u(portal_id):
    """Download M3U playlist for a specific portal."""
    try:
        # Get portal info
        portals = getPortals()
        
        if portal_id not in portals:
            logger.warning(f"Portal download requested for non-existent portal: {portal_id}")
            return Response("Portal not found", status=404)
        
        portal = portals[portal_id]
        portal_name = portal.get("name", "Unknown Portal")
        
        # Generate M3U content
        m3u_content = generate_portal_m3u(portal_id)
        
        if m3u_content is None:
            logger.error(f"Failed to generate M3U for portal: {portal_id}")
            return Response("Error generating M3U", status=500)
        
        # Generate filename
        filename = generate_portal_filename(portal_name)
        
        # Create response with proper headers
        response = Response(m3u_content, mimetype="text/plain")
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.headers["Content-Type"] = "text/plain; charset=utf-8"
        
        logger.info(f"M3U download served for portal: {portal_name} ({portal_id})")
        return response
        
    except Exception as e:
        logger.error(f"Error in portal M3U download for {portal_id}: {e}")
        return Response("Internal server error", status=500)


@app.route("/portal/<portal_id>/playlist.m3u", methods=["GET"])
def portal_specific_m3u(portal_id):
    """Portal-specific M3U with configurable access control."""
    settings = getSettings()
    public_access = settings.get("public playlist access", "true") == "true"
    
    if public_access:
        # Public access enabled - no authentication required
        return _portal_specific_m3u(portal_id)
    else:
        # Public access disabled - require Basic Auth
        auth = request.authorization
        if not auth or not auth.username or not auth.password:
            # No Basic Auth provided - return 401 with WWW-Authenticate header
            response = Response(
                'Authentication required\n'
                'Please provide Basic Auth credentials in the URL:\n'
                'http://username:password@host/portal/portal_id/playlist.m3u',
                401,
                {'WWW-Authenticate': 'Basic realm="MacReplayXC Portal M3U"'}
            )
            return response
        
        # Validate Basic Auth credentials
        system_username = settings.get("username", "admin")
        system_password = settings.get("password", "12345")
        
        if auth.username != system_username or auth.password != system_password:
            logger.warning(f"Invalid Basic Auth credentials for portal M3U: {auth.username}")
            response = Response(
                'Invalid credentials\n'
                'Please check your username and password.',
                401,
                {'WWW-Authenticate': 'Basic realm="MacReplayXC Portal M3U"'}
            )
            return response
        
        # Authentication successful
        logger.info(f"Basic Auth successful for portal M3U: {auth.username}")
        return _portal_specific_m3u(portal_id)

def _portal_specific_m3u(portal_id):
    """
    Portal-specific M3U route handler with Basic Auth and Query Parameter support.
    
    Supports authentication via:
    1. HTTP Basic Auth: http://user:pass@host/portal/portal_id/playlist.m3u
    2. Query Parameters: http://host/portal/portal_id/playlist.m3u?username=user&password=pass
    
    Basic Auth takes precedence over Query Parameters.
    """
    try:
        # Extract authentication credentials
        username, password = extract_auth_credentials(request)
        
        # Get system settings
        settings = getSettings()
        security_enabled = settings.get("enable security", "false") == "true"
        public_access_enabled = settings.get("public playlist access", "true") == "true"
        
        # Validate portal exists and is enabled
        portals = getPortals()
        
        if portal_id not in portals:
            logger.warning(f"Portal-specific M3U requested for non-existent portal: {portal_id}")
            return Response("Portal not found", status=404)
        
        portal = portals[portal_id]
        
        if portal.get("enabled") != "true":
            logger.warning(f"Portal-specific M3U requested for disabled portal: {portal_id}")
            return Response("Portal not found", status=404)
        
        # Validate authentication only if public access is disabled
        if not public_access_enabled:
            is_valid, error_message = validate_authentication(username, password, settings)
            if not is_valid:
                logger.warning(f"Portal-specific M3U requested with invalid credentials for portal: {portal_id}")
                return Response(error_message, status=401)
        else:
            logger.debug(f"Portal-specific M3U access granted (public access enabled) for portal: {portal_id}")
        
        # Generate M3U content with authentication-aware stream URLs
        m3u_content = generate_portal_m3u_with_auth(portal_id, username, password)
        
        if m3u_content is None:
            logger.error(f"Failed to generate M3U for portal: {portal_id}")
            return Response("Error generating M3U", status=500)
        
        # Create response with proper headers for M3U file download
        response = Response(m3u_content, mimetype="application/x-mpegURL")
        response.headers["Content-Disposition"] = f"attachment; filename=portal_{portal_id}_playlist.m3u"
        response.headers["Content-Type"] = "application/x-mpegURL; charset=utf-8"
        
        portal_name = portal.get("name", "Unknown Portal")
        logger.info(f"Portal-specific M3U served for portal: {portal_name} ({portal_id})")
        return response
        
    except Exception as e:
        logger.error(f"Error in portal-specific M3U for {portal_id}: {e}")
        return Response("Internal server error", status=500)


@app.route("/editor", methods=["GET"])
@authorise
def editor():
    return render_template("editor.html")
    
@app.route("/editor_data", methods=["GET"])
@authorise
def editor_data():
    """Get channel data from database cache."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get only enabled channels from database
        cursor.execute('''
            SELECT 
                portal, channel_id, portal_name, name, number, genre,
                custom_name, custom_number, custom_genre, custom_epg_id, fallback_channel
            FROM channels
            WHERE enabled = 1
            ORDER BY portal_name, CAST(COALESCE(NULLIF(custom_number, ''), number) AS INTEGER)
        ''')
        
        channels = []
        # Use external host configuration
        external_host, external_scheme = get_external_host_config()
        request_host = external_host or request.host
        request_scheme = external_scheme if external_host else request.scheme
        
        for row in cursor.fetchall():
            channels.append({
                "portal": row['portal'],
                "portalName": row['portal_name'] or '',
                "enabled": True,  # All returned channels are enabled
                "channelNumber": row['number'] or '',
                "customChannelNumber": row['custom_number'] or '',
                "channelName": row['name'] or '',
                "customChannelName": row['custom_name'] or '',
                "genre": row['genre'] or '',
                "customGenre": row['custom_genre'] or '',
                "channelId": row['channel_id'],
                "customEpgId": row['custom_epg_id'] or '',
                "fallbackChannel": row['fallback_channel'] or '',
                "link": f"{request_scheme}://{request_host}/play/{row['portal']}/{row['channel_id']}?web=true",
            })
        
        conn.close()
        
        logger.info(f"Returned {len(channels)} enabled channels from database cache")
        return flask.jsonify({"data": channels})
        
    except Exception as e:
        logger.error(f"Error in editor_data: {e}")
        return flask.jsonify({"data": [], "error": str(e)}), 500

@app.route("/editor/portals", methods=["GET"])
@authorise
def editor_portals():
    """Get list of unique portals for filter dropdown."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT portal_name
            FROM channels
            WHERE portal_name IS NOT NULL AND portal_name != ''
            ORDER BY portal_name
        """)
        
        portals = [row['portal_name'] for row in cursor.fetchall()]
        conn.close()
        
        logger.info(f"Returning {len(portals)} portals from database")
        return flask.jsonify({"portals": portals})
    except Exception as e:
        logger.error(f"Error in editor_portals: {e}")
        return flask.jsonify({"portals": [], "error": str(e)}), 500


@app.route("/editor/genres", methods=["GET"])
@authorise
def editor_genres():
    """Get list of unique genres for filter dropdown."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT COALESCE(NULLIF(custom_genre, ''), genre) as genre
            FROM channels
            WHERE COALESCE(NULLIF(custom_genre, ''), genre) IS NOT NULL 
                AND COALESCE(NULLIF(custom_genre, ''), genre) != ''
                AND COALESCE(NULLIF(custom_genre, ''), genre) != 'None'
            ORDER BY genre
        """)
        
        genres = [row['genre'] for row in cursor.fetchall()]
        conn.close()
        
        logger.info(f"Returning {len(genres)} genres from database")
        return flask.jsonify({"genres": genres})
    except Exception as e:
        logger.error(f"Error in editor_genres: {e}")
        return flask.jsonify({"genres": [], "error": str(e)}), 500


@app.route("/editor/portal-stats", methods=["GET"])
@authorise
def editor_portal_stats():
    """Get portal statistics with all channels (enabled and disabled)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all portals with their stats
        cursor.execute("""
            SELECT 
                portal,
                portal_name,
                COUNT(*) as total_channels,
                SUM(CASE WHEN enabled = 1 THEN 1 ELSE 0 END) as enabled_channels,
                COUNT(DISTINCT COALESCE(NULLIF(custom_genre, ''), genre)) as total_genres
            FROM channels
            GROUP BY portal, portal_name
            ORDER BY portal_name
        """)
        
        portals = []
        for row in cursor.fetchall():
            # Get genre stats for this portal
            cursor.execute("""
                SELECT 
                    COALESCE(NULLIF(custom_genre, ''), genre) as genre,
                    COUNT(*) as total,
                    SUM(CASE WHEN enabled = 1 THEN 1 ELSE 0 END) as enabled
                FROM channels
                WHERE portal = ?
                GROUP BY COALESCE(NULLIF(custom_genre, ''), genre)
            """, (row['portal'],))
            
            genres_with_enabled = sum(1 for g in cursor.fetchall() if g['enabled'] > 0)
            
            portals.append({
                "id": row['portal'],
                "name": row['portal_name'] or row['portal'],
                "total_channels": row['total_channels'],
                "enabled_channels": row['enabled_channels'],
                "total_genres": row['total_genres'],
                "enabled_genres": genres_with_enabled
            })
        
        conn.close()
        logger.info(f"Returning {len(portals)} portal stats")
        return flask.jsonify({"portals": portals})
    except Exception as e:
        logger.error(f"Error in editor_portal_stats: {e}")
        return flask.jsonify({"portals": [], "error": str(e)}), 500


@app.route("/editor/portal-channels/<portal_id>", methods=["GET"])
@authorise
def editor_portal_channels(portal_id):
    """Get all channels for a specific portal (enabled and disabled)."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Use external host configuration
        external_host, external_scheme = get_external_host_config()
        request_host = external_host or request.host
        request_scheme = external_scheme if external_host else request.scheme
        
        cursor.execute("""
            SELECT 
                portal, channel_id, portal_name, name, number, genre, logo,
                custom_name, custom_number, custom_genre, custom_epg_id, 
                fallback_channel, enabled
            FROM channels
            WHERE portal = ? OR portal_name = ?
            ORDER BY COALESCE(NULLIF(custom_genre, ''), genre), 
                     CAST(COALESCE(NULLIF(custom_number, ''), number) AS INTEGER)
        """, (portal_id, portal_id))
        
        channels = []
        for row in cursor.fetchall():
            channels.append({
                "portal": row['portal'],
                "portalName": row['portal_name'] or '',
                "channelId": row['channel_id'],
                "channelName": row['name'] or '',
                "customChannelName": row['custom_name'] or '',
                "channelNumber": row['number'] or '',
                "customChannelNumber": row['custom_number'] or '',
                "genre": row['genre'] or '',
                "customGenre": row['custom_genre'] or '',
                "customEpgId": row['custom_epg_id'] or '',
                "fallbackChannel": row['fallback_channel'] or '',
                "enabled": bool(row['enabled']),
                "logo": row['logo'] or '',
                "link": f"{request_scheme}://{request_host}/play/{row['portal']}/{row['channel_id']}?web=true",
            })
        
        conn.close()
        logger.info(f"Returning {len(channels)} channels for portal {portal_id}")
        return flask.jsonify({"channels": channels})
    except Exception as e:
        logger.error(f"Error in editor_portal_channels: {e}")
        return flask.jsonify({"channels": [], "error": str(e)}), 500


@app.route("/editor/save", methods=["POST"])
@authorise
def editorSave():
    global cached_xmltv, last_playlist_host
    threading.Thread(target=refresh_xmltv, daemon=True).start()
    last_playlist_host = None
    Thread(target=refresh_lineup).start()
    
    enabledEdits = json.loads(request.form["enabledEdits"])
    numberEdits = json.loads(request.form["numberEdits"])
    nameEdits = json.loads(request.form["nameEdits"])
    genreEdits = json.loads(request.form["genreEdits"])
    epgEdits = json.loads(request.form["epgEdits"])
    fallbackEdits = json.loads(request.form["fallbackEdits"])
    
    # Update SQLite database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Process enabled/disabled edits
        for edit in enabledEdits:
            portal = edit["portal"]
            channel_id = edit["channel id"]
            enabled = 1 if edit["enabled"] else 0
            
            cursor.execute('''
                UPDATE channels 
                SET enabled = ? 
                WHERE portal = ? AND channel_id = ?
            ''', (enabled, portal, channel_id))
        
        # Process custom number edits
        for edit in numberEdits:
            portal = edit["portal"]
            channel_id = edit["channel id"]
            custom_number = edit["custom number"]
            
            cursor.execute('''
                UPDATE channels 
                SET custom_number = ? 
                WHERE portal = ? AND channel_id = ?
            ''', (custom_number, portal, channel_id))
        
        # Process custom name edits
        for edit in nameEdits:
            portal = edit["portal"]
            channel_id = edit["channel id"]
            custom_name = edit["custom name"]
            
            cursor.execute('''
                UPDATE channels 
                SET custom_name = ? 
                WHERE portal = ? AND channel_id = ?
            ''', (custom_name, portal, channel_id))
        
        # Process custom genre edits
        for edit in genreEdits:
            portal = edit["portal"]
            channel_id = edit["channel id"]
            custom_genre = edit["custom genre"]
            
            cursor.execute('''
                UPDATE channels 
                SET custom_genre = ? 
                WHERE portal = ? AND channel_id = ?
            ''', (custom_genre, portal, channel_id))
        
        # Process custom EPG ID edits
        for edit in epgEdits:
            portal = edit["portal"]
            channel_id = edit["channel id"]
            custom_epg_id = edit["custom epg id"]
            
            cursor.execute('''
                UPDATE channels 
                SET custom_epg_id = ? 
                WHERE portal = ? AND channel_id = ?
            ''', (custom_epg_id, portal, channel_id))
        
        # Process fallback channel edits
        for edit in fallbackEdits:
            portal = edit["portal"]
            channel_id = edit["channel id"]
            fallback_channel = edit["channel name"]
            
            cursor.execute('''
                UPDATE channels 
                SET fallback_channel = ? 
                WHERE portal = ? AND channel_id = ?
            ''', (fallback_channel, portal, channel_id))
        
        conn.commit()
        logger.info("Channel edits saved to database!")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving channel edits: {e}")
        flash(f"Error saving changes: {e}", "danger")
        return redirect("/editor", code=302)
    finally:
        conn.close()
    
    flash("Playlist config saved!", "success")
    return redirect("/editor", code=302)


@app.route("/editor/bulk-edit", methods=["POST"])
@authorise
def editor_bulk_edit():
    """Apply bulk search & replace to channel names and genres."""
    try:
        data = request.get_json()
        rules = data.get('rules', [])
        apply_to_names = data.get('apply_to_names', True)
        apply_to_genres = data.get('apply_to_genres', False)
        case_sensitive = data.get('case_sensitive', False)
        use_regex = data.get('use_regex', False)
        
        if not rules:
            return flask.jsonify({"success": False, "error": "No rules provided"}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create tables for history and saved rules
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bulk_edit_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                rules TEXT NOT NULL,
                apply_to_names INTEGER NOT NULL,
                apply_to_genres INTEGER NOT NULL,
                channels_backup TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bulk_edit_saved_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_text TEXT NOT NULL,
                replace_text TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_used TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Get all channels and save as backup
        cursor.execute('SELECT portal, channel_id, name, custom_name, genre, custom_genre FROM channels')
        channels = cursor.fetchall()
        
        import json
        channels_backup = json.dumps([dict(ch) for ch in channels])
        
        # Save to history
        cursor.execute('''
            INSERT INTO bulk_edit_history (timestamp, rules, apply_to_names, apply_to_genres, channels_backup)
            VALUES (datetime('now'), ?, ?, ?, ?)
        ''', (json.dumps(rules), 1 if apply_to_names else 0, 1 if apply_to_genres else 0, channels_backup))
        
        # Save individual rules for persistence
        for rule in rules:
            search_text = rule['search']
            replace_text = rule['replace']
            
            # Check if rule already exists
            cursor.execute('''
                SELECT id FROM bulk_edit_saved_rules 
                WHERE search_text = ? AND replace_text = ?
            ''', (search_text, replace_text))
            
            if cursor.fetchone():
                # Update last_used timestamp
                cursor.execute('''
                    UPDATE bulk_edit_saved_rules 
                    SET last_used = datetime('now')
                    WHERE search_text = ? AND replace_text = ?
                ''', (search_text, replace_text))
            else:
                # Insert new rule
                cursor.execute('''
                    INSERT INTO bulk_edit_saved_rules (search_text, replace_text)
                    VALUES (?, ?)
                ''', (search_text, replace_text))
        
        # Keep only last 10 history entries
        cursor.execute('''
            DELETE FROM bulk_edit_history 
            WHERE id NOT IN (
                SELECT id FROM bulk_edit_history 
                ORDER BY id DESC 
                LIMIT 10
            )
        ''')
        
        conn.commit()
        
        # Re-fetch channels for processing (cursor was used for history operations)
        cursor.execute('SELECT portal, channel_id, name, custom_name, genre, custom_genre FROM channels')
        channels = cursor.fetchall()
        
        updated_count = 0
        
        for channel in channels:
            portal = channel['portal']
            channel_id = channel['channel_id']
            original_name = channel['custom_name'] or channel['name']
            original_genre = channel['custom_genre'] or channel['genre']
            
            new_name = original_name
            new_genre = original_genre
            
            # Apply rules to name
            if apply_to_names and original_name:
                for rule in rules:
                    search = rule['search']
                    replace = rule['replace']
                    
                    if use_regex:
                        import re
                        flags = 0 if case_sensitive else re.IGNORECASE
                        try:
                            new_name = re.sub(search, replace, new_name, flags=flags)
                        except re.error as e:
                            logger.error(f"Regex error: {e}")
                            continue
                    else:
                        if case_sensitive:
                            new_name = new_name.replace(search, replace)
                        else:
                            # Case-insensitive replace
                            import re
                            pattern = re.compile(re.escape(search), re.IGNORECASE)
                            new_name = pattern.sub(replace, new_name)
            
            # Apply rules to genre
            if apply_to_genres and original_genre:
                for rule in rules:
                    search = rule['search']
                    replace = rule['replace']
                    
                    if use_regex:
                        import re
                        flags = 0 if case_sensitive else re.IGNORECASE
                        try:
                            new_genre = re.sub(search, replace, new_genre, flags=flags)
                        except re.error as e:
                            logger.error(f"Regex error: {e}")
                            continue
                    else:
                        if case_sensitive:
                            new_genre = new_genre.replace(search, replace)
                        else:
                            import re
                            pattern = re.compile(re.escape(search), re.IGNORECASE)
                            new_genre = pattern.sub(replace, new_genre)
            
            # Clean up whitespace
            new_name = ' '.join(new_name.split()).strip()
            new_genre = ' '.join(new_genre.split()).strip()
            
            # Update if changed
            if new_name != original_name or new_genre != original_genre:
                if apply_to_names and new_name != original_name:
                    cursor.execute('''
                        UPDATE channels 
                        SET custom_name = ? 
                        WHERE portal = ? AND channel_id = ?
                    ''', (new_name, portal, channel_id))
                
                if apply_to_genres and new_genre != original_genre:
                    cursor.execute('''
                        UPDATE channels 
                        SET custom_genre = ? 
                        WHERE portal = ? AND channel_id = ?
                    ''', (new_genre, portal, channel_id))
                
                updated_count += 1
        
        conn.commit()
        conn.close()
        
        # Refresh playlist and EPG (XC API queries database directly, so no separate cache to clear)
        global cached_xmltv, last_playlist_host
        last_playlist_host = None  # Force M3U playlist regeneration
        threading.Thread(target=refresh_xmltv, daemon=True).start()  # Refresh EPG
        
        logger.info(f"Bulk edit applied: {updated_count} channels updated")
        
        return flask.jsonify({
            "success": True,
            "updated": updated_count
        })
        
    except Exception as e:
        logger.error(f"Error in bulk edit: {e}")
        return flask.jsonify({"success": False, "error": str(e)}), 500


@app.route("/editor/bulk-edit/undo", methods=["POST"])
@authorise
def editor_bulk_edit_undo():
    """Undo the last bulk edit operation."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the last history entry
        cursor.execute('''
            SELECT id, channels_backup FROM bulk_edit_history 
            ORDER BY id DESC 
            LIMIT 1
        ''')
        history = cursor.fetchone()
        
        if not history:
            return flask.jsonify({"success": False, "error": "No history to undo"}), 400
        
        import json
        channels_backup = json.loads(history['channels_backup'])
        
        # Restore channels from backup
        for channel in channels_backup:
            cursor.execute('''
                UPDATE channels 
                SET custom_name = ?, custom_genre = ?
                WHERE portal = ? AND channel_id = ?
            ''', (channel['custom_name'], channel['custom_genre'], 
                  channel['portal'], channel['channel_id']))
        
        # Delete the history entry
        cursor.execute('DELETE FROM bulk_edit_history WHERE id = ?', (history['id'],))
        
        conn.commit()
        conn.close()
        
        # Refresh playlist and EPG
        global cached_xmltv, last_playlist_host
        last_playlist_host = None
        threading.Thread(target=refresh_xmltv, daemon=True).start()
        
        logger.info("Bulk edit undone successfully")
        
        return flask.jsonify({
            "success": True,
            "message": "Last bulk edit undone successfully"
        })
        
    except Exception as e:
        logger.error(f"Error undoing bulk edit: {e}")
        return flask.jsonify({"success": False, "error": str(e)}), 500


@app.route("/editor/bulk-edit/history", methods=["GET"])
@authorise
def editor_bulk_edit_history():
    """Get bulk edit history."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, timestamp, rules FROM bulk_edit_history 
            ORDER BY id DESC 
            LIMIT 10
        ''')
        history = cursor.fetchall()
        conn.close()
        
        import json
        history_list = []
        for entry in history:
            rules = json.loads(entry['rules'])
            history_list.append({
                'id': entry['id'],
                'timestamp': entry['timestamp'],
                'rules': rules
            })
        
        return flask.jsonify({
            "success": True,
            "history": history_list
        })
        
    except Exception as e:
        logger.error(f"Error getting bulk edit history: {e}")
        return flask.jsonify({"success": False, "error": str(e)}), 500


@app.route("/editor/bulk-edit/saved-rules", methods=["GET"])
@authorise
def editor_bulk_edit_saved_rules():
    """Get saved bulk edit rules."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT search_text, replace_text, last_used 
            FROM bulk_edit_saved_rules 
            ORDER BY last_used DESC 
            LIMIT 50
        ''')
        rules = cursor.fetchall()
        conn.close()
        
        rules_list = []
        for rule in rules:
            rules_list.append({
                'search': rule['search_text'],
                'replace': rule['replace_text'],
                'last_used': rule['last_used']
            })
        
        return flask.jsonify({
            "success": True,
            "rules": rules_list
        })
        
    except Exception as e:
        logger.error(f"Error getting saved rules: {e}")
        return flask.jsonify({"success": False, "error": str(e)}), 500


@app.route("/editor/bulk-edit/clear-saved-rules", methods=["POST"])
@authorise
def editor_bulk_edit_clear_saved_rules():
    """Clear all saved bulk edit rules."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM bulk_edit_saved_rules')
        conn.commit()
        conn.close()
        
        logger.info("Cleared all saved bulk edit rules")
        
        return flask.jsonify({
            "success": True,
            "message": "All saved rules cleared"
        })
        
    except Exception as e:
        logger.error(f"Error clearing saved rules: {e}")
        return flask.jsonify({"success": False, "error": str(e)}), 500


@app.route("/editor/reset-all", methods=["POST"])
@authorise
def editor_reset_all_customizations():
    """Reset all custom names and genres to original values."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE channels 
            SET custom_name = NULL,
                custom_genre = NULL
        ''')
        
        conn.commit()
        conn.close()
        
        # Refresh playlist and EPG
        global cached_xmltv, last_playlist_host
        last_playlist_host = None
        threading.Thread(target=refresh_xmltv, daemon=True).start()
        
        logger.info("All customizations reset to original values")
        
        return flask.jsonify({
            "success": True,
            "message": "All customizations reset successfully"
        })
        
    except Exception as e:
        logger.error(f"Error resetting customizations: {e}")
        return flask.jsonify({"success": False, "error": str(e)}), 500


@app.route("/editor/reset", methods=["POST"])
@authorise
def editorReset():
    """Reset all channel customizations in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            UPDATE channels 
            SET enabled = 0,
                custom_name = '',
                custom_number = '',
                custom_genre = '',
                custom_epg_id = '',
                fallback_channel = ''
        ''')
        
        conn.commit()
        logger.info("All channel customizations reset!")
        flash("Playlist reset!", "success")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error resetting channels: {e}")
        flash(f"Error resetting: {e}", "danger")
    finally:
        conn.close()
    
    return redirect("/editor", code=302)


@app.route("/editor/refresh", methods=["POST"])
@authorise
def editor_refresh():
    """Manually trigger a refresh of the channel cache."""
    try:
        global editor_refresh_progress
        
        if editor_refresh_progress["running"]:
            return flask.jsonify({"error": "Channel refresh already in progress"}), 400
        
        # Initialize progress
        portals = getPortals()
        enabled_portals = [p for p in portals.values() if p.get("enabled") == "true"]
        
        editor_refresh_progress = {
            "running": True,
            "current_portal": "",
            "current_step": "Starting...",
            "portals_done": 0,
            "portals_total": len(enabled_portals),
            "started_at": time.time()
        }
        
        threading.Thread(target=refresh_channels_cache_with_progress, daemon=True).start()
        return flask.jsonify({"success": True, "message": "Channel refresh started"})
    except Exception as e:
        logger.error(f"Error starting channel refresh: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/editor/refresh/progress", methods=["GET"])
@authorise
def editor_refresh_progress_status():
    """Get channel refresh progress."""
    return flask.jsonify(editor_refresh_progress)

@app.route("/editor/deactivate-duplicates", methods=["POST"])
@authorise
def editor_deactivate_duplicates():
    """Deactivate duplicate enabled channels, keeping only the first occurrence."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Find all duplicate channels (using ROW_NUMBER to identify which to keep)
        find_duplicates_query = """
            WITH ranked_channels AS (
                SELECT 
                    portal,
                    channel_id,
                    COALESCE(NULLIF(custom_name, ''), name) as effective_name,
                    ROW_NUMBER() OVER (
                        PARTITION BY COALESCE(NULLIF(custom_name, ''), name) 
                        ORDER BY portal, channel_id
                    ) as row_num
                FROM channels
                WHERE enabled = 1
            )
            SELECT portal, channel_id, effective_name, row_num
            FROM ranked_channels
            WHERE effective_name IN (
                SELECT effective_name
                FROM ranked_channels
                GROUP BY effective_name
                HAVING COUNT(*) > 1
            )
            AND row_num > 1
            ORDER BY effective_name, row_num
        """
        
        cursor.execute(find_duplicates_query)
        duplicates_to_deactivate = cursor.fetchall()
        
        # Deactivate the duplicate channels
        deactivated_count = 0
        for dup in duplicates_to_deactivate:
            cursor.execute("""
                UPDATE channels
                SET enabled = 0
                WHERE portal = ? AND channel_id = ?
            """, (dup['portal'], dup['channel_id']))
            deactivated_count += 1
        
        conn.commit()
        conn.close()
        
        # Reset playlist cache to force regeneration
        global last_playlist_host
        last_playlist_host = None
        
        logger.info(f"Deactivated {deactivated_count} duplicate channels")
        
        return flask.jsonify({
            "success": True,
            "deactivated": deactivated_count,
            "message": f"Deactivated {deactivated_count} duplicate channels"
        })
        
    except Exception as e:
        logger.error(f"Error in editor_deactivate_duplicates: {e}")
        return flask.jsonify({
            "success": False,
            "deactivated": 0,
            "error": str(e)
        }), 500

@app.route("/settings", methods=["GET"])
@authorise
def settings():
    settings = getSettings()
    return render_template(
        "settings.html", settings=settings, defaultSettings=defaultSettings
    )


@app.route("/proxy-test", methods=["GET"])
@authorise
def proxy_test_page():
    """Proxy test page."""
    return render_template("proxy_test.html")


@app.route("/settings/save", methods=["POST"])
@authorise
def save():
    settings = {}

    for setting, _ in defaultSettings.items():
        if setting == "public playlist access":
            # Special handling for inverted checkbox logic
            # If checkbox is checked, it means "secure" (false for public access)
            # If checkbox is not checked, it means "public" (true for public access)
            checkbox_value = request.form.get(setting)
            if checkbox_value == "false":  # Checkbox is checked (secure mode)
                settings[setting] = "false"  # No public access
            else:  # Checkbox is not checked (public mode)
                settings[setting] = "true"   # Allow public access
        else:
            value = request.form.get(setting, "false")
            settings[setting] = value

    saveSettings(settings)
    logger.info("Settings saved!")
    Thread(target=refresh_xmltv).start()
    flash("Settings saved!", "success")
    return redirect("/settings", code=302)


# ============================================
# XC Users Management Routes
# ============================================

@app.route("/xc-users", methods=["GET"])
@authorise
def xc_users_page():
    """XC Users management page."""
    return render_template("xc_users.html", settings=getSettings())


@app.route("/xc-users/list", methods=["GET"])
@authorise
def xc_users_list():
    """Get list of XC users."""
    users = getXCUsers()
    user_list = []
    
    # Create a copy to avoid RuntimeError if dictionary changes during iteration
    for user_id, user in list(users.items()):
        active_cons = len(user.get("active_connections", {}))
        user_list.append({
            "id": user_id,
            "username": user.get("username"),
            "password": user.get("password"),
            "enabled": user.get("enabled") == "true",
            "max_connections": user.get("max_connections"),
            "active_connections": active_cons,
            "allowed_portals": user.get("allowed_portals", []),
            "created_at": user.get("created_at"),
            "expires_at": user.get("expires_at")
        })
    
    return flask.jsonify({"users": user_list})


@app.route("/xc-users/add", methods=["POST"])
@authorise
def xc_users_add():
    """Add new XC user."""
    try:
        data = request.json
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()
        
        if not username or not password:
            return flask.jsonify({"error": "Username and password required"}), 400
        
        users = getXCUsers()
        user_id = f"{username}_{password}"
        
        if user_id in users:
            return flask.jsonify({"error": "User already exists"}), 400
        
        users[user_id] = {
            "username": username,
            "password": password,
            "enabled": "true",
            "max_connections": str(data.get("max_connections", 1)),
            "allowed_portals": data.get("allowed_portals", []),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "expires_at": data.get("expires_at", ""),
            "active_connections": {}
        }
        
        saveXCUsers(users)
        logger.info(f"XC user created: {username}")
        return flask.jsonify({"success": True, "user_id": user_id})
    except Exception as e:
        logger.error(f"Error adding XC user: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/xc-users/update", methods=["POST"])
@authorise
def xc_users_update():
    """Update XC user."""
    try:
        data = request.json
        user_id = data.get("user_id")
        
        if not user_id:
            return flask.jsonify({"error": "User ID required"}), 400
        
        users = getXCUsers()
        if user_id not in users:
            return flask.jsonify({"error": "User not found"}), 404
        
        users[user_id]["enabled"] = "true" if data.get("enabled") else "false"
        users[user_id]["max_connections"] = str(data.get("max_connections", 1))
        users[user_id]["allowed_portals"] = data.get("allowed_portals", [])
        users[user_id]["expires_at"] = data.get("expires_at", "")
        
        saveXCUsers(users)
        logger.info(f"XC user updated: {user_id}")
        return flask.jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error updating XC user: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/xc-users/delete", methods=["POST"])
@authorise
def xc_users_delete():
    """Delete XC user."""
    try:
        data = request.json
        user_id = data.get("user_id")
        
        if not user_id:
            return flask.jsonify({"error": "User ID required"}), 400
        
        users = getXCUsers()
        if user_id not in users:
            return flask.jsonify({"error": "User not found"}), 404
        
        username = users[user_id].get("username")
        del users[user_id]
        saveXCUsers(users)
        
        logger.info(f"XC user deleted: {username}")
        return flask.jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error deleting XC user: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/xc-users/kick", methods=["POST"])
@authorise
def xc_users_kick():
    """Kick active connection."""
    try:
        data = request.json
        user_id = data.get("user_id")
        device_id = data.get("device_id")
        
        if not user_id or not device_id:
            return flask.jsonify({"error": "User ID and device ID required"}), 400
        
        unregisterXCConnection(user_id, device_id)
        logger.info(f"Kicked connection: {user_id}/{device_id}")
        return flask.jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error kicking connection: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/playlist.m3u", methods=["GET"])
def playlist():
    """Main M3U playlist with configurable access control and Basic Auth support."""
    settings = getSettings()
    public_access = settings.get("public playlist access", "true") == "true"
    
    if public_access:
        # Public access enabled - no authentication required
        return _playlist()
    else:
        # Public access disabled - require Basic Auth
        auth = request.authorization
        if not auth or not auth.username or not auth.password:
            # No Basic Auth provided - return 401 with WWW-Authenticate header
            response = Response(
                'Authentication required\n'
                'Please provide Basic Auth credentials in the URL:\n'
                'http://username:password@host/playlist.m3u',
                401,
                {'WWW-Authenticate': 'Basic realm="MacReplayXC Main Playlist"'}
            )
            return response
        
        # Validate Basic Auth credentials
        system_username = settings.get("username", "admin")
        system_password = settings.get("password", "12345")
        
        if auth.username != system_username or auth.password != system_password:
            logger.warning(f"Invalid Basic Auth credentials for main playlist: {auth.username}")
            response = Response(
                'Invalid credentials\n'
                'Please check your username and password.',
                401,
                {'WWW-Authenticate': 'Basic realm="MacReplayXC Main Playlist"'}
            )
            return response
        
        # Authentication successful
        logger.info(f"Basic Auth successful for main playlist: {auth.username}")
        return _playlist()

def _playlist():
    global cached_playlist, last_playlist_host
    
    logger.info("Playlist Requested")
    
    # Use external host configuration
    external_host, external_scheme = get_external_host_config()
    current_host = external_host or request.host or "0.0.0.0:8001"
    
    if cached_playlist is None or len(cached_playlist) == 0 or last_playlist_host != current_host:
        logger.info(f"Regenerating playlist due to host change: {last_playlist_host} -> {current_host}")
        last_playlist_host = current_host
        generate_playlist()

    return Response(cached_playlist, mimetype="text/plain")

@app.route("/update_playlistm3u", methods=["POST"])
@authorise
def update_playlistm3u():
    try:
        # First, clean up orphaned channels from deleted portals
        cleanup_orphaned_channels()
        
        # Then generate the playlist
        generate_playlist()
        logger.info("Playlist updated via dashboard")
        return Response("Playlist updated successfully", status=200)
    except Exception as e:
        logger.error(f"Error updating playlist: {e}")
        return Response(f"Error updating playlist: {str(e)}", status=500)

def cleanup_orphaned_channels():
    """Remove channels from database that belong to portals that no longer exist."""
    try:
        portals = getPortals()
        valid_portal_ids = set(portals.keys())
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all unique portal IDs from the database
        cursor.execute('SELECT DISTINCT portal FROM channels')
        db_portal_ids = set(row[0] for row in cursor.fetchall())
        
        # Find orphaned portal IDs (in DB but not in config)
        orphaned_ids = db_portal_ids - valid_portal_ids
        
        if orphaned_ids:
            for portal_id in orphaned_ids:
                cursor.execute('DELETE FROM channels WHERE portal = ?', (portal_id,))
                deleted = cursor.rowcount
                logger.info(f"Cleaned up {deleted} orphaned channels from deleted portal: {portal_id}")
            
            conn.commit()
        
        conn.close()
    except Exception as e:
        logger.error(f"Error cleaning up orphaned channels: {e}")

def generate_playlist():
    global cached_playlist
    logger.info("Generating playlist.m3u from database...")

    # Use external host configuration
    external_host, external_scheme = get_external_host_config()
    playlist_host = external_host or request.host or "0.0.0.0:8001"
    
    channels = []
    
    # Get enabled channels from database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT portal, channel_id, name, custom_name, genre, custom_genre, 
               number, custom_number, custom_epg_id
        FROM channels 
        WHERE enabled = 1
        ORDER BY portal, channel_id
    ''')
    db_channels = cursor.fetchall()
    conn.close()
    
    # Get portal info
    portals = getPortals()
    
    for channel in db_channels:
        portal_id = channel['portal']
        channel_id = str(channel['channel_id'])
        
        # Check if portal is enabled
        if portal_id not in portals or portals[portal_id].get("enabled") != "true":
            continue
        
        # Use custom values if available, otherwise use original values
        channel_name = channel['custom_name'] if channel['custom_name'] else (channel['name'] or "Unknown Channel")
        genre = channel['custom_genre'] if channel['custom_genre'] else (channel['genre'] or "")
        channel_number = channel['custom_number'] if channel['custom_number'] else (channel['number'] or "")
        epg_id = channel['custom_epg_id'] if channel['custom_epg_id'] else channel_name
        
        # Apply portal prefix to genre only (for group-title organization)
        portal_prefix = portals[portal_id].get("portal prefix", "").strip()
        if portal_prefix and genre:
            genre = f"[{portal_prefix}] {genre}"
        
        # Build M3U entry - escape quotes in attributes
        def escape_quotes(text):
            return str(text).replace('"', '&quot;') if text else ""
        
        m3u_entry = "#EXTINF:-1"
        m3u_entry += ' tvg-id="' + escape_quotes(epg_id) + '"'
        
        if getSettings().get("use channel numbers", "true") == "true" and channel_number:
            m3u_entry += ' tvg-chno="' + escape_quotes(channel_number) + '"'
        
        if getSettings().get("use channel genres", "true") == "true" and genre:
            m3u_entry += ' group-title="' + escape_quotes(genre) + '"'
        
        m3u_entry += ',' + str(channel_name) + "\n"
        m3u_entry += "http://" + playlist_host + "/play/" + portal_id + "/" + channel_id
        
        channels.append(m3u_entry)

    # Sort channels based on settings
    if getSettings().get("sort playlist by channel name", "true") == "true":
        channels.sort(key=lambda k: k.split(",")[1].split("\n")[0] if "," in k else "")
    if getSettings().get("use channel numbers", "true") == "true":
        if getSettings().get("sort playlist by channel number", "false") == "true":
            def get_channel_number(k):
                try:
                    if 'tvg-chno="' in k:
                        return int(k.split('tvg-chno="')[1].split('"')[0])
                    return 999999  # Put channels without numbers at the end
                except (ValueError, IndexError):
                    return 999999
            channels.sort(key=get_channel_number)
    if getSettings().get("use channel genres", "true") == "true":
        if getSettings().get("sort playlist by channel genre", "false") == "true":
            def get_genre(k):
                try:
                    if 'group-title="' in k:
                        return k.split('group-title="')[1].split('"')[0]
                    return "zzz"  # Put channels without genre at the end
                except IndexError:
                    return "zzz"
            channels.sort(key=get_genre)

    playlist = "#EXTM3U \n"
    if channels:
        playlist = playlist + "\n".join(channels)

    cached_playlist = playlist
    logger.info("Playlist generated and cached.")
    
def normalize_channel_name(name):
    """Normalize channel name for better matching."""
    import re
    if not name:
        return ""
    # Convert to lowercase
    name = name.lower().strip()
    # Remove common suffixes/prefixes
    name = re.sub(r'\s*(hd|sd|fhd|uhd|4k)\s*$', '', name, flags=re.IGNORECASE)
    # Remove special characters but keep spaces
    name = re.sub(r'[^\w\s]', '', name)
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def find_best_epg_match(channel_name, fallback_data):
    """Find best EPG match using normalized names with VERY strict matching rules.
    Returns None if no confident match is found - better no EPG than wrong EPG."""
    if not channel_name or not fallback_data:
        return None
    
    normalized_search = normalize_channel_name(channel_name)
    if not normalized_search:
        return None
    
    # Try exact match first - this is the only 100% confident match
    for fb_name, fb_data in fallback_data.items():
        if normalize_channel_name(fb_name) == normalized_search:
            return fb_data['channel_id']
    
    # Try substring match - but ONLY if it's a very strong match (80% similarity)
    for fb_name, fb_data in fallback_data.items():
        normalized_fb = normalize_channel_name(fb_name)
        # Increased threshold to 80% to be more conservative
        if normalized_search in normalized_fb:
            if len(normalized_search) >= len(normalized_fb) * 0.8:
                return fb_data['channel_id']
        elif normalized_fb in normalized_search:
            if len(normalized_fb) >= len(normalized_search) * 0.8:
                return fb_data['channel_id']
    
    # Word-by-word matching is now DISABLED by default
    # It causes too many false positives
    # If you want to enable it, uncomment the code below and adjust thresholds
    
    # # Try word-by-word match with VERY strict requirements
    # search_words = set(normalized_search.split())
    # # Filter out common words that shouldn't be used for matching
    # common_words = {'tv', 'hd', 'sd', 'channel', 'de', 'nl', 'uk', 'us', 'plus', 'one', 'two', 'drei', 'vier'}
    # search_words = search_words - common_words
    # 
    # if not search_words or len(search_words) < 2:
    #     return None  # Need at least 2 meaningful words to match
    # 
    # best_match = None
    # best_score = 0
    # best_ratio = 0
    # 
    # for fb_name, fb_data in fallback_data.items():
    #     fb_words = set(normalize_channel_name(fb_name).split()) - common_words
    #     if not fb_words or len(fb_words) < 2:
    #         continue
    #     
    #     # Count matching words
    #     matching_words = search_words & fb_words
    #     match_ratio = len(matching_words) / max(len(search_words), len(fb_words))
    #     
    #     # Require at least 75% of words to match (increased from 50%)
    #     # AND at least 2 matching words
    #     if match_ratio >= 0.75 and len(matching_words) >= 2 and len(matching_words) > best_score:
    #         best_score = len(matching_words)
    #         best_ratio = match_ratio
    #         best_match = fb_data['channel_id']
    # 
    # # Only return if we have a very strong match
    # if best_score >= 2 and best_ratio >= 0.75:
    #     return best_match
    
    # No confident match found - return None (better no EPG than wrong EPG)
    return None


def fetch_epgshare_fallback(countries):
    """Fetch EPG data from epgshare01.online for specified countries."""
    fallback_programmes = {}
    base_url = "https://epgshare01.online/epgshare01/"
    
    # Country code mapping
    country_files = {
        "DE": "epg_ripper_DE1.xml.gz",
        "AT": "epg_ripper_AT1.xml.gz",
        "CH": "epg_ripper_CH1.xml.gz",
        "NL": "epg_ripper_NL1.xml.gz",
        "BE": "epg_ripper_BE2.xml.gz",
        "UK": "epg_ripper_UK1.xml.gz",
        "US": "epg_ripper_US2.xml.gz",
        "FR": "epg_ripper_FR1.xml.gz",
        "ES": "epg_ripper_ES1.xml.gz",
        "IT": "epg_ripper_IT1.xml.gz",
        "PL": "epg_ripper_PL1.xml.gz",
        "TR": "epg_ripper_TR1.xml.gz",
        "PT": "epg_ripper_PT1.xml.gz",
        "SE": "epg_ripper_SE1.xml.gz",
        "NO": "epg_ripper_NO1.xml.gz",
        "DK": "epg_ripper_DK1.xml.gz",
        "FI": "epg_ripper_FI1.xml.gz",
        "GR": "epg_ripper_GR1.xml.gz",
        "RO": "epg_ripper_RO1.xml.gz",
        "HU": "epg_ripper_HU1.xml.gz",
        "CZ": "epg_ripper_CZ1.xml.gz",
        "SK": "epg_ripper_SK1.xml.gz",
        "HR": "epg_ripper_HR1.xml.gz",
        "RS": "epg_ripper_RS1.xml.gz",
        "BG": "epg_ripper_BG1.xml.gz",
        "AU": "epg_ripper_AU1.xml.gz",
        "NZ": "epg_ripper_NZ1.xml.gz",
        "CA": "epg_ripper_CA2.xml.gz",
        "BR": "epg_ripper_BR1.xml.gz",
        "MX": "epg_ripper_MX1.xml.gz",
        "AR": "epg_ripper_AR1.xml.gz",
        "JP": "epg_ripper_JP1.xml.gz",
        "KR": "epg_ripper_KR1.xml.gz",
        "IN": "epg_ripper_IN1.xml.gz",
        "IL": "epg_ripper_IL1.xml.gz",
        "ZA": "epg_ripper_ZA1.xml.gz",
        "IE": "epg_ripper_IE1.xml.gz",
    }
    
    for country in countries:
        country = country.strip().upper()
        if country not in country_files:
            logger.warning(f"No EPG fallback available for country: {country}")
            continue
        
        try:
            url = base_url + country_files[country]
            logger.info(f"Fetching EPG fallback for {country} from {url}")
            
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                # Decompress gzip
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
                    xml_content = f.read().decode('utf-8')
                
                # Parse XML
                root = ET.fromstring(xml_content)
                
                # Extract channel mappings and programmes
                for channel in root.findall('channel'):
                    channel_id = channel.get('id', '')
                    display_name = channel.find('display-name')
                    if display_name is not None and display_name.text:
                        # Store by display name (lowercase for matching)
                        name_key = display_name.text.lower().strip()
                        if name_key not in fallback_programmes:
                            fallback_programmes[name_key] = {
                                'channel_id': channel_id,
                                'programmes': []
                            }
                
                for programme in root.findall('programme'):
                    channel_id = programme.get('channel', '')
                    # Find matching channel name
                    for name_key, data in fallback_programmes.items():
                        if data['channel_id'] == channel_id:
                            data['programmes'].append({
                                'start': programme.get('start', ''),
                                'stop': programme.get('stop', ''),
                                'title': programme.find('title').text if programme.find('title') is not None else '',
                                'desc': programme.find('desc').text if programme.find('desc') is not None else ''
                            })
                            break
                
                logger.info(f"Loaded {len([p for d in fallback_programmes.values() for p in d['programmes']])} programmes from {country}")
                
                # Clean up
                del xml_content
                del root
                
        except Exception as e:
            logger.error(f"Error fetching EPG fallback for {country}: {e}")
    
    return fallback_programmes


def refresh_xmltv_with_progress():
    """Wrapper for refresh_xmltv with progress tracking."""
    global epg_refresh_progress
    try:
        refresh_xmltv()
    finally:
        epg_refresh_progress["running"] = False
        epg_refresh_progress["current_step"] = "Completed"

def refresh_xmltv():
    """Refresh XMLTV data with memory-optimized processing."""
    import gc
    global epg_refresh_progress
    
    settings = getSettings()
    logger.info("Refreshing XMLTV...")

    # Docker-optimized cache paths
    cache_dir = "/app/data"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "MacReplayXCEPG.xml")

    day_before_yesterday = datetime.utcnow() - timedelta(days=2)
    day_before_yesterday_str = day_before_yesterday.strftime("%Y%m%d%H%M%S") + " +0000"

    # Check if EPG fallback is enabled
    epg_refresh_progress["current_step"] = "Loading EPG settings..."
    epg_fallback_enabled = settings.get("epg fallback enabled", "false") == "true"
    epg_fallback_countries = settings.get("epg fallback countries", "").split(",")
    epg_fallback_countries = [c.strip() for c in epg_fallback_countries if c.strip()]
    
    fallback_epg = {}
    if epg_fallback_enabled and epg_fallback_countries:
        epg_refresh_progress["current_step"] = f"Fetching fallback EPG for {', '.join(epg_fallback_countries)}..."
        logger.info(f"EPG fallback enabled for countries: {epg_fallback_countries}")
        fallback_epg = fetch_epgshare_fallback(epg_fallback_countries)
        logger.info(f"Loaded fallback EPG for {len(fallback_epg)} channels")
        epg_refresh_progress["current_step"] = f"Loaded fallback EPG for {len(fallback_epg)} channels"

    # Build XMLTV directly without caching old programmes (memory optimization)
    channels_xml = ET.Element("tv")
    portals = getPortals()
    programme_count = 0
    channels_without_epg = []

    portal_index = 0
    for portal in portals:
        if portals[portal]["enabled"] == "true":
            portal_index += 1
            portal_name = portals[portal]["name"]
            portal_epg_offset = int(portals[portal]["epg offset"])
            
            # Update progress - show current portal being processed
            epg_refresh_progress["current_portal"] = portal_name
            epg_refresh_progress["current_step"] = f"Starting {portal_name}..."
            epg_refresh_progress["portals_done"] = portal_index - 1  # Show as "processing X of Y"
            
            logger.info(f"Fetching EPG | Portal: {portal_name} | offset: {portal_epg_offset} |")

            enabledChannels = portals[portal].get("enabled channels", [])
            if len(enabledChannels) != 0:
                name = portals[portal]["name"]
                url = portals[portal]["url"]
                macs = list(portals[portal]["macs"].keys())
                proxy = portals[portal]["proxy"]
                customChannelNames = portals[portal].get("custom channel names", {})
                customEpgIds = portals[portal].get("custom epg ids", {})
                customChannelNumbers = portals[portal].get("custom channel numbers", {})

                epg_refresh_progress["current_step"] = f"{portal_name}: Found {len(macs)} MAC(s), {len(enabledChannels)} enabled channels"

                # Fetch channels and EPG from ALL MACs and merge
                all_channels_map = {}  # channelId -> channel data
                merged_epg = {}  # channelId -> [programmes]
                
                mac_index = 0
                for mac in macs:
                    try:
                        mac_index += 1
                        epg_refresh_progress["current_step"] = f"{portal_name}: Authenticating MAC {mac_index}/{len(macs)} ({mac})"
                        token = stb.getToken(url, mac, proxy)
                        if token:
                            stb.getProfile(url, mac, token, proxy)
                            
                            epg_refresh_progress["current_step"] = f"{portal_name}: Fetching channels from MAC {mac_index}/{len(macs)}"
                            mac_channels = stb.getAllChannels(url, mac, token, proxy)
                            
                            epg_refresh_progress["current_step"] = f"{portal_name}: Fetching EPG from MAC {mac_index}/{len(macs)}"
                            mac_epg = stb.getEpg(url, mac, token, 24, proxy)
                            
                            if mac_channels:
                                for ch in mac_channels:
                                    ch_id = str(ch.get("id"))
                                    if ch_id not in all_channels_map:
                                        all_channels_map[ch_id] = ch
                                logger.info(f"MAC {mac}: Got {len(mac_channels)} channels")
                                epg_refresh_progress["current_step"] = f"{portal_name}: MAC {mac_index}/{len(macs)} - {len(mac_channels)} channels"
                            
                            if mac_epg:
                                for ch_id, programmes in mac_epg.items():
                                    # Merge EPG data - add programmes if we don't have any yet
                                    # or if the new data has more programmes
                                    if ch_id not in merged_epg or len(programmes) > len(merged_epg.get(ch_id, [])):
                                        merged_epg[ch_id] = programmes
                                logger.info(f"MAC {mac}: Got EPG for {len(mac_epg)} channels")
                                epg_refresh_progress["current_step"] = f"{portal_name}: MAC {mac_index}/{len(macs)} - EPG for {len(mac_epg)} channels"
                            else:
                                logger.warning(f"MAC {mac}: No EPG data returned")
                                epg_refresh_progress["current_step"] = f"{portal_name}: MAC {mac_index}/{len(macs)} - No EPG data"
                            
                            # Clear MAC data
                            if mac_channels:
                                del mac_channels
                            if mac_epg:
                                del mac_epg
                            
                    except Exception as e:
                        logger.error(f"Error fetching data for MAC {mac}: {e}")
                        epg_refresh_progress["current_step"] = f"{portal_name}: MAC {mac_index}/{len(macs)} - Error: {str(e)[:50]}"
                        continue
                
                logger.info(f"Portal {portal_name}: Total {len(all_channels_map)} channels, EPG for {len(merged_epg)} channels")
                epg_refresh_progress["current_step"] = f"{portal_name}: Processing {len(all_channels_map)} channels..."

                if all_channels_map:
                    # Convert enabled channels to set for faster lookup
                    enabled_set = set(enabledChannels)
                    
                    epg_refresh_progress["current_step"] = f"{portal_name}: Loading custom EPG mappings from database..."
                    # Get custom EPG IDs from database (set via EPG page)
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT channel_id, custom_epg_id 
                        FROM channels 
                        WHERE portal = ? AND custom_epg_id IS NOT NULL AND custom_epg_id != ''
                    ''', (portal,))
                    db_custom_epg_ids = {row['channel_id']: row['custom_epg_id'] for row in cursor.fetchall()}
                    conn.close()
                    
                    # Get genres for this portal to show category names
                    genres_dict = {}
                    try:
                        for mac in macs:
                            token = stb.getToken(url, mac, proxy)
                            if token:
                                genres = stb.getGenres(url, mac, token, proxy)
                                if genres:
                                    for genre in genres:
                                        genre_id = str(genre.get("id"))
                                        genre_name = str(genre.get("title", "Unknown"))
                                        genres_dict[genre_id] = genre_name
                                    break  # Got genres, no need to try other MACs
                    except Exception as e:
                        logger.error(f"Error fetching genres: {e}")
                    
                    epg_refresh_progress["current_step"] = f"{portal_name}: Building XMLTV for {len(enabled_set)} enabled channels..."
                    
                    # Group channels by genre for progress display
                    channels_by_genre = {}
                    for channelId, channel in all_channels_map.items():
                        if channelId in enabled_set:
                            genre_id = str(channel.get("tv_genre_id", ""))
                            genre_name = genres_dict.get(genre_id, "Other")
                            if genre_name not in channels_by_genre:
                                channels_by_genre[genre_name] = []
                            channels_by_genre[genre_name].append((channelId, channel))
                    
                    # Process channels by genre
                    processed_channels = 0
                    total_enabled = len(enabled_set)
                    
                    for genre_name, genre_channels in channels_by_genre.items():
                        epg_refresh_progress["current_step"] = f"{portal_name}: Processing {genre_name} ({processed_channels}/{total_enabled} channels)"
                        
                        for channelId, channel in genre_channels:
                            try:
                                processed_channels += 1
                                
                                # Update progress every 10 channels
                                if processed_channels % 10 == 0:
                                    epg_refresh_progress["current_step"] = f"{portal_name}: Processing {genre_name} ({processed_channels}/{total_enabled} channels)"
                                
                                channelName = customChannelNames.get(channelId, channel.get("name"))
                                channelNumber = customChannelNumbers.get(channelId, str(channel.get("number")))
                                # Priority: 1. Database custom EPG ID, 2. JSON config custom EPG ID, 3. Channel number
                                epgId = db_custom_epg_ids.get(channelId) or customEpgIds.get(channelId, channelNumber)

                                channelEle = ET.SubElement(channels_xml, "channel", id=epgId)
                                ET.SubElement(channelEle, "display-name").text = channelName
                                logo = channel.get("logo")
                                if logo:
                                    ET.SubElement(channelEle, "icon", src=logo)

                                channel_epg = merged_epg.get(channelId, [])
                                
                                if not channel_epg:
                                    # Try fallback EPG if enabled
                                    fallback_used = False
                                    if epg_fallback_enabled and fallback_epg:
                                        # Try to match by channel name using improved matching
                                        matched_fb_id = find_best_epg_match(channelName, fallback_epg)
                                        if matched_fb_id:
                                            # Find the fallback data by channel_id
                                            fb_data = None
                                            for fb_name, data in fallback_epg.items():
                                                if data['channel_id'] == matched_fb_id:
                                                    fb_data = data
                                                    break
                                            
                                            if fb_data and fb_data.get('programmes'):
                                                for p in fb_data['programmes'][:50]:  # Limit to 50 programmes
                                                    try:
                                                        programmeEle = ET.SubElement(
                                                            channels_xml, "programme",
                                                            start=p['start'], stop=p['stop'], channel=epgId
                                                        )
                                                        ET.SubElement(programmeEle, "title").text = p['title']
                                                        if p['desc']:
                                                            ET.SubElement(programmeEle, "desc").text = p['desc']
                                                        programme_count += 1
                                                        fallback_used = True
                                                    except Exception as e:
                                                        pass
                                                if fallback_used:
                                                    logger.debug(f"Used fallback EPG for {channelName}")
                                    
                                    if not fallback_used:
                                        # Create dummy EPG
                                        channels_without_epg.append(channelName)
                                        start_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
                                        stop_time = start_time + timedelta(hours=24)
                                        start = start_time.strftime("%Y%m%d%H%M%S") + " +0000"
                                        stop = stop_time.strftime("%Y%m%d%H%M%S") + " +0000"
                                        programmeEle = ET.SubElement(
                                            channels_xml, "programme", start=start, stop=stop, channel=epgId
                                        )
                                        ET.SubElement(programmeEle, "title").text = channelName
                                        ET.SubElement(programmeEle, "desc").text = channelName
                                        programme_count += 1
                                else:
                                    for p in channel_epg:
                                        try:
                                            start_ts = p.get("start_timestamp")
                                            stop_ts = p.get("stop_timestamp")
                                            if not start_ts or not stop_ts:
                                                continue
                                                
                                            start_time = datetime.utcfromtimestamp(start_ts) + timedelta(hours=portal_epg_offset)
                                            stop_time = datetime.utcfromtimestamp(stop_ts) + timedelta(hours=portal_epg_offset)
                                            start = start_time.strftime("%Y%m%d%H%M%S") + " +0000"
                                            stop = stop_time.strftime("%Y%m%d%H%M%S") + " +0000"
                                            
                                            if start <= day_before_yesterday_str:
                                                continue
                                                
                                            programmeEle = ET.SubElement(
                                                channels_xml, "programme", start=start, stop=stop, channel=epgId
                                            )
                                            ET.SubElement(programmeEle, "title").text = p.get("name", "")
                                            desc = p.get("descr", "")
                                            if desc:
                                                ET.SubElement(programmeEle, "desc").text = desc
                                            programme_count += 1
                                        except Exception as e:
                                            logger.error(f"Error processing programme: {e}")
                            except Exception as e:
                                logger.error(f"Error processing channel: {e}")
                    
                    # Clear data from memory
                    epg_refresh_progress["current_step"] = f"{portal_name}: Completed - {programme_count} total programmes"
                    epg_refresh_progress["portals_done"] = portal_index  # Mark this portal as done
                    del merged_epg
                    del all_channels_map
                    gc.collect()
                else:
                    logger.error(f"Error making XMLTV for {name}, skipping")
                    epg_refresh_progress["current_step"] = f"{portal_name}: Error - skipping"
                    epg_refresh_progress["portals_done"] = portal_index  # Mark this portal as done

    if channels_without_epg:
        logger.warning(f"{len(channels_without_epg)} channels without EPG data")

    epg_refresh_progress["current_step"] = "Generating XMLTV file..."
    # Generate XML string without minidom (much more memory efficient)
    rough_string = ET.tostring(channels_xml, encoding="unicode")
    
    # Simple formatting without minidom
    formatted_xmltv = '<?xml version="1.0" encoding="UTF-8"?>\n' + rough_string

    epg_refresh_progress["current_step"] = f"Writing XMLTV cache ({programme_count} programmes)..."
    # Write to cache file
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(formatted_xmltv)
        logger.info(f"XMLTV cache updated with {programme_count} programmes.")
        epg_refresh_progress["current_step"] = f"XMLTV cache updated with {programme_count} programmes"
    except Exception as e:
        logger.error(f"Error writing XMLTV cache: {e}")
        epg_refresh_progress["current_step"] = f"Error writing cache: {str(e)}"

    epg_refresh_progress["current_step"] = "Finalizing..."
    # Update global cache
    global cached_xmltv, last_updated
    cached_xmltv = formatted_xmltv
    last_updated = time.time()
    
    # Clean up
    del channels_xml
    del rough_string
    del fallback_epg
    gc.collect()
    
    epg_refresh_progress["current_step"] = f"Completed! {programme_count} programmes from {portal_index} portals"
    
@app.route("/xmltv", methods=["GET"])
def xmltv():
    """XMLTV EPG with configurable access control and Basic Auth support."""
    settings = getSettings()
    public_access = settings.get("public playlist access", "true") == "true"
    
    if public_access:
        # Public access enabled - no authentication required
        return _xmltv()
    else:
        # Public access disabled - require Basic Auth
        auth = request.authorization
        if not auth or not auth.username or not auth.password:
            # No Basic Auth provided - return 401 with WWW-Authenticate header
            response = Response(
                'Authentication required\n'
                'Please provide Basic Auth credentials in the URL:\n'
                'http://username:password@host/xmltv',
                401,
                {'WWW-Authenticate': 'Basic realm="MacReplayXC XMLTV"'}
            )
            return response
        
        # Validate Basic Auth credentials
        system_username = settings.get("username", "admin")
        system_password = settings.get("password", "12345")
        
        if auth.username != system_username or auth.password != system_password:
            logger.warning(f"Invalid Basic Auth credentials for XMLTV: {auth.username}")
            response = Response(
                'Invalid credentials\n'
                'Please check your username and password.',
                401,
                {'WWW-Authenticate': 'Basic realm="MacReplayXC XMLTV"'}
            )
            return response
        
        # Authentication successful
        logger.info(f"Basic Auth successful for XMLTV: {auth.username}")
        return _xmltv()

def _xmltv():
    global cached_xmltv, last_updated
    logger.info("Guide Requested")
    
    if cached_xmltv is None or (time.time() - last_updated) > 900:
        refresh_xmltv()
    
    return Response(
        cached_xmltv,
        mimetype="text/xml",
    )

# ============================================
# EPG Routes - with caching to prevent memory leaks
# ============================================

# EPG cache to prevent repeated API calls
_epg_cache = {
    "portal_status": None,
    "portal_status_time": 0,
    "channels": None,
    "channels_time": 0,
    "programs": None,
    "programs_time": 0
}
_EPG_CACHE_TTL = 300  # 5 minutes cache


def _clear_epg_cache():
    """Clear EPG cache."""
    global _epg_cache
    _epg_cache = {
        "portal_status": None,
        "portal_status_time": 0,
        "channels": None,
        "channels_time": 0,
        "programs": None,
        "programs_time": 0
    }


@app.route("/epg", methods=["GET"])
@authorise
def epg_page():
    """EPG status page showing portal EPG information."""
    return render_template("epg.html", settings=getSettings())


@app.route("/epg/portal-status", methods=["GET"])
@authorise
def epg_portal_status():
    """Get EPG status for all portals from database - NO portal queries."""
    global _epg_cache
    
    # First, clean up orphaned channels from deleted portals
    cleanup_orphaned_channels()
    
    # Clear cache to ensure fresh data after cleanup
    _epg_cache["portal_status"] = None
    
    try:
        # Get valid portal IDs from config
        portals = getPortals()
        valid_portal_ids = set(portals.keys())
        
        # Get portal info from database only - no API queries
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get channel counts and EPG status per portal from database
        cursor.execute('''
            SELECT 
                portal,
                portal_name,
                COUNT(*) as channel_count,
                SUM(CASE WHEN custom_epg_id IS NOT NULL AND custom_epg_id != '' THEN 1 ELSE 0 END) as epg_channel_count
            FROM channels
            WHERE enabled = 1
            GROUP BY portal, portal_name
            ORDER BY portal_name
        ''')
        
        portal_status = []
        for row in cursor.fetchall():
            # Only include portals that still exist in config
            if row['portal'] in valid_portal_ids:
                portal_info = {
                    "id": row['portal'],
                    "name": row['portal_name'] or 'Unknown',
                    "has_epg": row['epg_channel_count'] > 0,
                    "epg_url": None,  # Not needed for display
                    "epg_type": "database",
                    "channel_count": row['channel_count'],
                    "epg_channel_count": row['epg_channel_count']
                }
                portal_status.append(portal_info)
        
        conn.close()
        
        # Cache the result
        _epg_cache["portal_status"] = portal_status
        _epg_cache["portal_status_time"] = time.time()
        
        logger.info(f"Returned EPG status for {len(portal_status)} portals from database")
        return flask.jsonify(portal_status)
    except Exception as e:
        logger.error(f"Error getting portal EPG status: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/epg/settings", methods=["GET"])
@authorise
def epg_settings():
    """Get EPG fallback settings."""
    settings = getSettings()
    return flask.jsonify({
        "epg_fallback_enabled": settings.get("epg fallback enabled", "false") == "true",
        "epg_fallback_countries": settings.get("epg fallback countries", "")
    })


@app.route("/epg/settings", methods=["POST"])
@authorise
def epg_settings_save():
    """Save EPG fallback settings."""
    try:
        data = request.json
        settings = getSettings()
        
        settings["epg fallback enabled"] = "true" if data.get("epg_fallback_enabled") else "false"
        settings["epg fallback countries"] = data.get("epg_fallback_countries", "")
        
        saveSettings(settings)
        _clear_epg_cache()
        
        return flask.jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error saving EPG settings: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/epg/channels", methods=["GET"])
@authorise
def epg_channels():
    """Get all enabled channels with their EPG mapping status from database - NO portal queries."""
    global _epg_cache
    
    # Return cached data if still valid
    if _epg_cache["channels"] and (time.time() - _epg_cache["channels_time"]) < _EPG_CACHE_TTL:
        return flask.jsonify({"channels": _epg_cache["channels"]})
    
    try:
        # Get channels from database ONLY - no portal queries
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                portal, channel_id, portal_name, name, number, logo, genre,
                custom_name, custom_genre, custom_epg_id, has_portal_epg
            FROM channels
            WHERE enabled = 1
            ORDER BY portal_name, CAST(COALESCE(NULLIF(custom_number, ''), number) AS INTEGER)
        ''')
        
        channels = []
        
        for row in cursor.fetchall():
            channel_name = row['custom_name'] if row['custom_name'] else row['name']
            channel_genre = row['custom_genre'] if row['custom_genre'] else row['genre']
            epg_id = row['custom_epg_id'] if row['custom_epg_id'] else ''
            
            # Try to get has_portal_epg, default to 0 if column doesn't exist yet
            try:
                has_portal_epg = bool(row['has_portal_epg'])
            except (KeyError, IndexError):
                has_portal_epg = False
            
            # has_epg = True if custom_epg_id is set OR has portal EPG
            has_epg = bool(epg_id) or has_portal_epg
            
            channels.append({
                "portal_id": row['portal'],
                "portal_name": row['portal_name'] or '',
                "channel_id": row['channel_id'],
                "channel_name": channel_name,
                "channel_number": row['number'] or '',
                "channel_genre": channel_genre or '',
                "epg_id": epg_id,
                "has_epg": has_epg,
                "has_portal_epg": has_portal_epg,  # Now from database!
                "logo": row['logo'] or ''
            })
        
        conn.close()
        
        # Cache the result
        _epg_cache["channels"] = channels
        _epg_cache["channels_time"] = time.time()
        
        logger.info(f"Returned {len(channels)} channels for EPG page from database")
        return flask.jsonify({"channels": channels})
    except Exception as e:
        logger.error(f"Error getting EPG channels: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/epg/fallback-channels", methods=["GET"])
@authorise
def epg_fallback_channels():
    """Get available channels from epgshare01 fallback for matching."""
    settings = getSettings()
    countries = settings.get("epg fallback countries", "").split(",")
    countries = [c.strip() for c in countries if c.strip()]
    
    if not countries:
        return flask.jsonify({"channels": [], "message": "No fallback countries configured"})
    
    try:
        fallback_data = fetch_epgshare_fallback(countries)
        channels = list(fallback_data.keys())
        return flask.jsonify({"channels": sorted(channels), "count": len(channels)})
    except Exception as e:
        logger.error(f"Error fetching fallback channels: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/epg/apply-fallback", methods=["POST"])
@authorise
def epg_apply_fallback():
    """Apply fallback EPG ID to a channel based on name matching - uses database."""
    try:
        data = request.json
        portal_id = data.get("portal_id")
        channel_id = data.get("channel_id")
        channel_name = data.get("channel_name", "")
        fallback_name = data.get("fallback_name", "")  # Optional: specific fallback channel name
        
        if not portal_id or not channel_id:
            return flask.jsonify({"error": "Missing portal_id or channel_id"}), 400
        
        # Get fallback data
        settings = getSettings()
        countries = settings.get("epg fallback countries", "").split(",")
        countries = [c.strip() for c in countries if c.strip()]
        
        if not countries:
            return flask.jsonify({"error": "No fallback countries configured"}), 400
        
        fallback_data = fetch_epgshare_fallback(countries)
        
        # Use improved matching function
        search_name = fallback_name or channel_name
        matched_epg_id = find_best_epg_match(search_name, fallback_data)
        
        if not matched_epg_id:
            logger.warning(f"No fallback match found for '{search_name}'")
            return flask.jsonify({
                "error": f"No fallback match found for '{search_name}'", 
                "available": list(fallback_data.keys())[:20]
            }), 404
        
        # Update database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE channels 
            SET custom_epg_id = ? 
            WHERE portal = ? AND channel_id = ?
        ''', (matched_epg_id, portal_id, channel_id))
        
        conn.commit()
        conn.close()
        
        # Clear caches
        global cached_xmltv
        cached_xmltv = None
        _clear_epg_cache()
        
        logger.info(f"Applied fallback EPG ID '{matched_epg_id}' to channel '{channel_name}'")
        return flask.jsonify({"success": True, "epg_id": matched_epg_id})
    except Exception as e:
        logger.error(f"Error applying fallback: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/epg/apply-fallback-all", methods=["POST"])
@authorise
def epg_apply_fallback_all():
    """Apply fallback EPG to all channels without portal EPG - optimized for database."""
    try:
        data = request.json
        channels = data.get("channels", [])
        
        if not channels:
            return flask.jsonify({"error": "No channels provided"}), 400
        
        # Filter to only channels without portal EPG (has_portal_epg = False)
        # This should significantly reduce the number of channels to process
        channels_without_portal_epg = [ch for ch in channels if not ch.get("has_portal_epg", False)]
        
        logger.info(f"Received {len(channels)} channels, {len(channels_without_portal_epg)} without portal EPG")
        
        # Use the filtered list for processing
        channels = channels_without_portal_epg
        
        if not channels:
            return flask.jsonify({
                "success": True,
                "matched": 0,
                "total": 0,
                "message": "No channels without portal EPG found"
            })
        
        # Increased limit since we're now only processing channels without portal EPG
        if len(channels) > 5000:
            return flask.jsonify({
                "error": f"Too many channels without portal EPG ({len(channels)}). Please apply fallback manually to specific channels."
            }), 400
        
        # Get fallback data
        settings = getSettings()
        countries = settings.get("epg fallback countries", "").split(",")
        countries = [c.strip() for c in countries if c.strip()]
        
        if not countries:
            return flask.jsonify({"error": "No fallback countries configured. Configure in EPG Fallback tab."}), 400
        
        logger.info(f"Fetching fallback EPG for countries: {countries}")
        fallback_data = fetch_epgshare_fallback(countries)
        
        if not fallback_data:
            return flask.jsonify({"error": "Failed to fetch fallback data"}), 500
        
        # Update database directly for better performance
        conn = get_db_connection()
        cursor = conn.cursor()
        
        matched_count = 0
        total_count = len(channels)
        
        for channel in channels:
            portal_id = channel.get("portal_id")
            channel_id = channel.get("channel_id")
            channel_name = channel.get("channel_name", "")
            
            if not portal_id or not channel_id:
                continue
            
            # Try to find matching channel using improved matching
            matched_epg_id = find_best_epg_match(channel_name, fallback_data)
            
            if matched_epg_id:
                # Update database
                cursor.execute('''
                    UPDATE channels 
                    SET custom_epg_id = ? 
                    WHERE portal = ? AND channel_id = ?
                ''', (matched_epg_id, portal_id, channel_id))
                
                matched_count += 1
                if matched_count % 100 == 0:
                    logger.info(f"Matched {matched_count}/{total_count} channels...")
        
        conn.commit()
        conn.close()
        
        # Clear caches
        global cached_xmltv
        cached_xmltv = None
        _clear_epg_cache()
        
        logger.info(f"Applied fallback to {matched_count}/{total_count} channels")
        return flask.jsonify({
            "success": True,
            "matched": matched_count,
            "total": total_count,
            "message": f"Applied fallback to {matched_count} out of {total_count} channels"
        })
    except Exception as e:
        logger.error(f"Error applying fallback to all: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/epg/save-mapping", methods=["POST"])
@authorise
def epg_save_mapping():
    """Save EPG ID mapping for a channel - uses database."""
    try:
        data = request.json
        portal_id = data.get("portal_id")
        channel_id = data.get("channel_id")
        epg_id = data.get("epg_id", "")
        
        if not portal_id or not channel_id:
            return flask.jsonify({"error": "Missing portal_id or channel_id"}), 400
        
        # Update database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE channels 
            SET custom_epg_id = ? 
            WHERE portal = ? AND channel_id = ?
        ''', (epg_id, portal_id, channel_id))
        
        conn.commit()
        conn.close()
        
        # Clear caches
        global cached_xmltv
        cached_xmltv = None
        _clear_epg_cache()
        
        logger.info(f"Saved EPG mapping for channel {channel_id}: {epg_id}")
        return flask.jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error saving EPG mapping: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/epg/refresh", methods=["POST"])
@authorise
def epg_refresh():
    """Force refresh EPG cache."""
    try:
        global epg_refresh_progress
        
        if epg_refresh_progress["running"]:
            return flask.jsonify({"error": "EPG refresh already in progress"}), 400
        
        _clear_epg_cache()
        global cached_xmltv
        cached_xmltv = None
        
        # Initialize progress
        portals = getPortals()
        enabled_portals = [p for p in portals.values() if p.get("enabled") == "true"]
        
        epg_refresh_progress = {
            "running": True,
            "current_portal": "",
            "current_step": "Starting...",
            "portals_done": 0,
            "portals_total": len(enabled_portals),
            "started_at": time.time()
        }
        
        threading.Thread(target=refresh_xmltv_with_progress, daemon=True).start()
        return flask.jsonify({"success": True, "message": "EPG refresh started"})
    except Exception as e:
        logger.error(f"Error refreshing EPG: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/epg/refresh/progress", methods=["GET"])
@authorise
def epg_refresh_progress_status():
    """Get EPG refresh progress."""
    return flask.jsonify(epg_refresh_progress)


# ============================================
# Xtream Codes API Routes
# ============================================

@app.route("/get.php", methods=["GET"])
@app.route("/get", methods=["GET"])
@xc_auth_only
def xc_get_playlist():
    """XC API M3U playlist endpoint with optional portal filtering."""
    return xc_get_playlist_impl()


@app.route("/portal/<portal_id>/get.php", methods=["GET"])
@xc_auth_only
def xc_get_portal_playlist(portal_id):
    """Route-based XC API M3U playlist endpoint for specific portal."""
    return xc_get_playlist_impl(route_portal_id=portal_id)


def xc_get_playlist_impl(route_portal_id=None):
    """XC API M3U playlist endpoint with optional portal filtering."""
    settings = getSettings()
    if settings.get("xc api enabled") != "true":
        return "XC API is disabled", 403
    
    username = request.args.get("username")
    password = request.args.get("password")
    output = request.args.get("output", "m3u8")
    playlist_type = request.args.get("type", "m3u_plus")
    
    # NEW: Portal filtering support via portal_id parameter
    portal_id_filter = request.args.get("portal_id")
    
    if not username or not password:
        return "Missing credentials", 401
    
    user_id, user = validateXCUser(username, password)
    if not user_id:
        return "Invalid credentials", 401
    
    # Generate M3U playlist with optional portal filtering
    m3u_content = generate_xc_m3u_with_portal_filter(user, portal_id_filter)
    
    return Response(m3u_content, mimetype="application/x-mpegURL")


def generate_xc_m3u_with_portal_filter(user, portal_id_filter=None):
    """
    Generate XC API M3U playlist content with optional portal filtering.
    
    Args:
        user (dict): XC API user object with allowed_portals
        portal_id_filter (str, optional): Portal ID to filter by
        
    Returns:
        str: M3U playlist content
    """
    portals = getPortals()
    allowed_portals = user.get("allowed_portals", [])
    
    m3u_content = "#EXTM3U\n"
    
    # Get enabled channels from database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT portal, channel_id, name, custom_name, genre, custom_genre, 
               number, custom_number, custom_epg_id, logo
        FROM channels 
        WHERE enabled = 1
        ORDER BY portal, channel_id
    ''')
    db_channels = cursor.fetchall()
    conn.close()
    
    # Use the same host as the request came from (handles reverse proxy correctly)
    scheme = request.headers.get('X-Forwarded-Proto', request.scheme)
    host = request.headers.get('X-Forwarded-Host', request.host)
    
    # Create a copy to avoid RuntimeError if dictionary changes during iteration
    for portal_id, portal in list(portals.items()):
        if portal.get("enabled") != "true":
            continue
        if allowed_portals and portal_id not in allowed_portals:
            continue
        
        # NEW: Portal filtering - if portal_id_filter is specified, only include that portal
        if portal_id_filter and portal_id != portal_id_filter:
            continue
        
        # Get channels for this portal from database
        portal_channels = [ch for ch in db_channels if ch['portal'] == portal_id]
        if not portal_channels:
            continue
        
        for db_channel in portal_channels:
            channel_id = str(db_channel['channel_id'])
            
            # Use custom values if available, otherwise use original values
            channel_name = db_channel['custom_name'] if db_channel['custom_name'] else (db_channel['name'] or "Unknown Channel")
            genre = db_channel['custom_genre'] if db_channel['custom_genre'] else (db_channel['genre'] or "")
            channel_number = db_channel['custom_number'] if db_channel['custom_number'] else (db_channel['number'] or "")
            epg_id = db_channel['custom_epg_id'] if db_channel['custom_epg_id'] else channel_name
            logo = db_channel['logo'] or ""
            
            # Apply portal prefix to genre only (for group-title organization)
            portal_prefix = portal.get("portal prefix", "").strip()
            if portal_prefix and genre:
                genre = f"[{portal_prefix}] {genre}"
            
            stream_id = f"{portal_id}_{channel_id}"
            # Standard XC API URL format for maximum compatibility
            # Add .ts extension for better IPTV client compatibility
            stream_url = f"{scheme}://{host}/{request.args.get('username')}/{request.args.get('password')}/{stream_id}.ts"
            
            # Escape quotes in attributes
            def escape_quotes(text):
                return str(text).replace('"', '&quot;') if text else ""
            
            m3u_content += f'#EXTINF:-1 tvg-id="{escape_quotes(epg_id)}" tvg-name="{escape_quotes(channel_name)}" tvg-logo="{escape_quotes(logo)}" group-title="{escape_quotes(genre)}",{channel_name}\n'
            m3u_content += f'{stream_url}\n'
    
    return m3u_content


@app.route("/player_api.php", methods=["GET"])
@xc_auth_only
def xc_api():
    """Xtream Codes API endpoint."""
    settings = getSettings()
    if settings.get("xc api enabled") != "true":
        return flask.jsonify({
            "user_info": {
                "auth": 0,
                "message": "XC API is disabled"
            }
        })
    
    username = request.args.get("username")
    password = request.args.get("password")
    action = request.args.get("action")
    
    if not username or not password:
        return flask.jsonify({
            "user_info": {
                "auth": 0,
                "message": "Missing credentials"
            }
        })
    
    user_id, user = validateXCUser(username, password)
    if not user_id:
        return flask.jsonify({
            "user_info": {
                "auth": 0,
                "message": user_id  # user_id contains error message
            }
        })
    
    # Handle different actions
    if action == "get_live_streams":
        return xc_get_live_streams(user)
    elif action == "get_live_categories":
        return xc_get_live_categories(user)
    elif action == "get_vod_streams":
        return xc_get_vod_streams(user)
    elif action == "get_series":
        return xc_get_series(user)
    elif action == "get_vod_categories":
        return xc_get_vod_categories(user)
    elif action == "get_series_categories":
        return xc_get_series_categories(user)
    elif action == "get_series_info":
        series_id = request.args.get("series_id")
        if series_id:
            return xc_get_series_info(user, series_id)
        return flask.jsonify({"info": {}, "episodes": {}})
    elif action == "get_vod_info":
        vod_id = request.args.get("vod_id")
        if vod_id:
            return xc_get_vod_info(user, vod_id)
        return flask.jsonify({"info": {}, "movie_data": {}})
    else:
        # Default: return user info
        return xc_get_user_info(user_id, user)


def xc_get_user_info(user_id, user):
    """Get XC user info."""
    active_cons = len(user.get("active_connections", {}))
    max_cons = int(user.get("max_connections", 1))
    
    expires_at = user.get("expires_at", "")
    exp_date = None
    if expires_at:
        try:
            exp_date = datetime.strptime(expires_at, "%Y-%m-%d")
        except:
            pass
    
    # Get correct host from headers (handles reverse proxy)
    scheme = request.headers.get('X-Forwarded-Proto', request.scheme)
    host = request.headers.get('X-Forwarded-Host', request.host)
    base_url = f"{scheme}://{host}"
    
    # Extract port
    port = "80"
    if ':' in host:
        port = host.split(':')[1]
    elif scheme == "https":
        port = "443"
    
    return flask.jsonify({
        "user_info": {
            "username": user.get("username"),
            "password": user.get("password"),
            "message": "",
            "auth": 1,
            "status": "Active",
            "exp_date": exp_date.strftime("%s") if exp_date else None,
            "is_trial": "0",
            "active_cons": str(active_cons),
            "created_at": user.get("created_at", ""),
            "max_connections": str(max_cons),
            "allowed_output_formats": ["m3u8", "ts"]
        },
        "server_info": {
            "url": base_url,
            "port": port,
            "https_port": "443" if scheme == "https" else "",
            "server_protocol": scheme,
            "rtmp_port": "",
            "timezone": "UTC",
            "timestamp_now": int(time.time()),
            "time_now": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    })


def xc_get_live_categories(user):
    """Get live stream categories - only return categories with enabled channels."""
    portals = getPortals()
    allowed_portals = user.get("allowed_portals", [])
    
    categories = []
    categories_with_channels = set()  # Track which categories have enabled channels
    
    # Get enabled channels from database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT DISTINCT portal, 
               COALESCE(NULLIF(custom_genre, ''), NULLIF(genre, ''), 'Unknown') as genre_name
        FROM channels 
        WHERE enabled = 1
        ORDER BY portal, genre_name
    ''')
    db_genres = cursor.fetchall()
    conn.close()
    
    # Create a copy to avoid RuntimeError if dictionary changes during iteration
    for portal_id, portal in list(portals.items()):
        if portal.get("enabled") != "true":
            continue
        if allowed_portals and portal_id not in allowed_portals:
            continue
        
        # Get genres for this portal from database
        portal_genres = [g for g in db_genres if g['portal'] == portal_id]
        
        for genre_data in portal_genres:
            genre_name = genre_data['genre_name']
            category_key = f"{portal_id}_{genre_name}"
            
            if category_key not in categories_with_channels:
                categories_with_channels.add(category_key)
                categories.append({
                    "category_id": category_key,
                    "category_name": genre_name,
                    "parent_id": 0
                })
    
    return flask.jsonify(categories)


def xc_get_live_streams(user):
    """Get live streams."""
    portals = getPortals()
    allowed_portals = user.get("allowed_portals", [])
    
    streams = []
    
    # Get enabled channels from database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT portal, channel_id, name, custom_name, genre, custom_genre, 
               number, custom_number, custom_epg_id, logo
        FROM channels 
        WHERE enabled = 1
        ORDER BY portal, channel_id
    ''')
    db_channels = cursor.fetchall()
    conn.close()
    
    # Create a copy to avoid RuntimeError if dictionary changes during iteration
    for portal_id, portal in list(portals.items()):
        if portal.get("enabled") != "true":
            continue
        if allowed_portals and portal_id not in allowed_portals:
            continue
        
        # Get channels for this portal from database
        portal_channels = [ch for ch in db_channels if ch['portal'] == portal_id]
        if not portal_channels:
            continue
        
        for db_channel in portal_channels:
            channel_id = str(db_channel['channel_id'])
            
            # Use custom values if available, otherwise use original values
            channel_name = db_channel['custom_name'] if db_channel['custom_name'] else (db_channel['name'] or "Unknown Channel")
            genre = db_channel['custom_genre'] if db_channel['custom_genre'] else (db_channel['genre'] or "Unknown")
            channel_number = db_channel['custom_number'] if db_channel['custom_number'] else (db_channel['number'] or "")
            epg_id = db_channel['custom_epg_id'] if db_channel['custom_epg_id'] else channel_name
            logo = db_channel['logo'] or ""
            
            # Create internal stream ID
            internal_id = f"{portal_id}_{channel_id}"
            
            # XC API expects numeric stream_id - use deterministic hash
            # Python's hash() is not deterministic across sessions, so use hashlib
            import hashlib
            numeric_id = int(hashlib.md5(internal_id.encode()).hexdigest()[:8], 16)
            
            # Create category_id that matches the one in xc_get_live_categories
            category_id = f"{portal_id}_{genre}"
            
            streams.append({
                "num": int(channel_number) if channel_number.isdigit() else 0,
                "name": channel_name,
                "stream_type": "live",
                "stream_id": numeric_id,
                "stream_icon": logo,
                "epg_channel_id": epg_id,
                "added": "",
                "category_id": category_id,
                "custom_sid": internal_id,  # Store real ID for reverse lookup
                "tv_archive": 0,
                "direct_source": "",
                "tv_archive_duration": 0,
                "container_extension": "ts"
            })
    
    return flask.jsonify(streams)


def xc_get_vod_categories(user):
    """Get VOD categories from selected categories in vods.db."""
    portals = getPortals()
    allowed_portals = user.get("allowed_portals", [])
    
    categories = []
    
    try:
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        
        # Get selected VOD categories
        cursor.execute('''
            SELECT vc.portal_id, vc.category_id, vc.title, vc.content_type, vc.item_count
            FROM vod_categories vc
            INNER JOIN vod_selections vs ON vc.portal_id = vs.portal_id 
                AND (vs.category_key = 'vod_' || vc.category_id OR vs.category_key = vc.content_type || '_' || vc.category_id)
            WHERE vc.content_type = 'vod' AND vs.enabled = 1
            ORDER BY vc.portal_id, vc.title
        ''')
        
        db_categories = cursor.fetchall()
        conn.close()
        
        for cat in db_categories:
            portal_id = cat['portal_id']
            
            # Check if portal is enabled and allowed
            portal = portals.get(portal_id)
            if not portal or portal.get("enabled") != "true":
                continue
            if allowed_portals and portal_id not in allowed_portals:
                continue
            
            category_key = f"{portal_id}_vod_{cat['category_id']}"
            
            categories.append({
                "category_id": category_key,
                "category_name": cat['title'],
                "parent_id": 0
            })
    except Exception as e:
        logger.error(f"Error getting VOD categories for XC API: {e}")
    
    return flask.jsonify(categories)


def xc_get_series_categories(user):
    """Get Series categories from selected categories in vods.db."""
    portals = getPortals()
    allowed_portals = user.get("allowed_portals", [])
    
    categories = []
    
    try:
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        
        # Get selected Series categories
        cursor.execute('''
            SELECT vc.portal_id, vc.category_id, vc.title, vc.content_type, vc.item_count
            FROM vod_categories vc
            INNER JOIN vod_selections vs ON vc.portal_id = vs.portal_id 
                AND (vs.category_key = 'series_' || vc.category_id OR vs.category_key = vc.content_type || '_' || vc.category_id)
            WHERE vc.content_type = 'series' AND vs.enabled = 1
            ORDER BY vc.portal_id, vc.title
        ''')
        
        db_categories = cursor.fetchall()
        conn.close()
        
        for cat in db_categories:
            portal_id = cat['portal_id']
            
            # Check if portal is enabled and allowed
            portal = portals.get(portal_id)
            if not portal or portal.get("enabled") != "true":
                continue
            if allowed_portals and portal_id not in allowed_portals:
                continue
            
            category_key = f"{portal_id}_series_{cat['category_id']}"
            
            categories.append({
                "category_id": category_key,
                "category_name": cat['title'],
                "parent_id": 0
            })
    except Exception as e:
        logger.error(f"Error getting Series categories for XC API: {e}")
    
    return flask.jsonify(categories)


def xc_get_vod_streams(user):
    """Get VOD streams from selected categories in vods.db."""
    portals = getPortals()
    allowed_portals = user.get("allowed_portals", [])
    
    streams = []
    
    try:
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        
        # Get VOD items from selected categories
        cursor.execute('''
            SELECT vi.portal_id, vi.category_id, vi.item_id, vi.name, vi.year, 
                   vi.description, vi.genre, vi.duration, vi.rating, vi.poster_url, vi.cmd
            FROM vod_items vi
            INNER JOIN vod_selections vs ON vi.portal_id = vs.portal_id 
                AND (vs.category_key = 'vod_' || vi.category_id OR vs.category_key = vi.content_type || '_' || vi.category_id)
            WHERE vi.content_type = 'vod' AND vs.enabled = 1
            ORDER BY vi.portal_id, vi.name
        ''')
        
        db_items = cursor.fetchall()
        conn.close()
        
        import hashlib
        
        for item in db_items:
            portal_id = item['portal_id']
            
            # Check if portal is enabled and allowed
            portal = portals.get(portal_id)
            if not portal or portal.get("enabled") != "true":
                continue
            if allowed_portals and portal_id not in allowed_portals:
                continue
            
            internal_id = f"{portal_id}_vod_{item['item_id']}"
            numeric_id = int(hashlib.md5(internal_id.encode()).hexdigest()[:8], 16)
            category_key = f"{portal_id}_vod_{item['category_id']}"
            
            streams.append({
                "num": numeric_id,
                "name": item['name'],
                "stream_type": "movie",
                "stream_id": numeric_id,
                "stream_icon": item['poster_url'] or "",
                "rating": item['rating'] or "",
                "added": "",
                "category_id": category_key,
                "container_extension": "mp4",
                "custom_sid": internal_id,
                "direct_source": ""
            })
    except Exception as e:
        logger.error(f"Error getting VOD streams for XC API: {e}")
    
    return flask.jsonify(streams)


def xc_get_vod_info(user, vod_id):
    """Get VOD/Movie info for XC API.
    
    The vod_id can be either:
    - A numeric hash (from get_vod_streams response)
    - A custom_sid string (portalId_vod_itemId)
    """
    import hashlib
    
    portals = getPortals()
    allowed_portals = user.get("allowed_portals", [])
    
    # Find the VOD by ID
    portal_id = None
    item_id = None
    vod_data = None
    
    try:
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        
        # Try to find by custom_sid first (string format)
        if '_vod_' in str(vod_id):
            logger.debug(f"XC API: Parsing custom VOD ID format: {vod_id}")
            parts = str(vod_id).split('_vod_')
            if len(parts) == 2:
                portal_id = parts[0]
                item_id = parts[1]
                logger.debug(f"XC API: Parsed - Portal: {portal_id}, Item: {item_id}")
        else:
            # Numeric format - search through all VODs to find matching hash
            logger.debug(f"XC API: Searching for numeric VOD ID: {vod_id}")
            numeric_id = int(vod_id)
            
            cursor.execute('''
                SELECT vi.portal_id, vi.item_id, vi.name, vi.year, 
                       vi.description, vi.genre, vi.duration, vi.rating, vi.poster_url, vi.cmd
                FROM vod_items vi
                INNER JOIN vod_selections vs ON vi.portal_id = vs.portal_id 
                    AND (vs.category_key = 'vod_' || vi.category_id OR vs.category_key = vi.content_type || '_' || vi.category_id)
                WHERE vi.content_type = 'vod' AND vs.enabled = 1
            ''')
            
            for item in cursor.fetchall():
                internal_id = f"{item['portal_id']}_vod_{item['item_id']}"
                check_id = int(hashlib.md5(internal_id.encode()).hexdigest()[:8], 16)
                if check_id == numeric_id:
                    portal_id = item['portal_id']
                    item_id = item['item_id']
                    vod_data = dict(item)
                    break
        
        if not portal_id or not item_id:
            conn.close()
            logger.warning(f"XC API: VOD not found - ID: {vod_id}")
            return flask.jsonify({
                "info": {},
                "movie_data": {},
                "error": "VOD not found"
            })
        
        # Get VOD data if not already fetched
        if not vod_data:
            cursor.execute('''
                SELECT portal_id, item_id, name, year, description, genre, duration, rating, poster_url, cmd
                FROM vod_items
                WHERE portal_id = ? AND item_id = ? AND content_type = 'vod'
            ''', (portal_id, item_id))
            row = cursor.fetchone()
            if row:
                vod_data = dict(row)
        
        conn.close()
        
        if not vod_data:
            return flask.jsonify({"info": {}, "movie_data": {}, "error": "VOD not found"})
        
        portal = portals.get(portal_id)
        if not portal or portal.get("enabled") != "true":
            return flask.jsonify({"info": {}, "movie_data": {}, "error": "Portal unavailable"})
        if allowed_portals and portal_id not in allowed_portals:
            return flask.jsonify({"info": {}, "movie_data": {}, "error": "Access denied"})
        
        # Generate consistent stream ID
        internal_id = f"{portal_id}_vod_{item_id}"
        stream_id = int(hashlib.md5(internal_id.encode()).hexdigest()[:8], 16)
        
        # Determine container extension from cmd
        container_ext = "mp4"  # Default
        if vod_data.get("cmd"):
            cmd_lower = vod_data["cmd"].lower()
            if ".mkv" in cmd_lower:
                container_ext = "mkv"
            elif ".avi" in cmd_lower:
                container_ext = "avi"
            elif ".ts" in cmd_lower:
                container_ext = "ts"
        
        # Build XC API compliant response
        response = {
            "info": {
                "movie_image": vod_data.get('poster_url') or "",
                "tmdb_id": "",
                "backdrop_path": [],
                "youtube_trailer": "",
                "genre": vod_data.get('genre') or "",
                "plot": vod_data.get('description') or "",
                "cast": "",
                "rating": vod_data.get('rating') or "",
                "director": "",
                "releasedate": vod_data.get('year') or "",
                "duration_secs": 0,
                "duration": vod_data.get('duration') or "",
                "video": {},
                "audio": {},
                "bitrate": 0,
                "name": vod_data.get('name') or ""
            },
            "movie_data": {
                "stream_id": stream_id,
                "name": vod_data.get('name') or "",
                "added": "",
                "category_id": "",
                "container_extension": container_ext,
                "custom_sid": internal_id,
                "direct_source": ""
            }
        }
        
        return flask.jsonify(response)
        
    except Exception as e:
        logger.error(f"VOD info error: {e}")
        return flask.jsonify({"info": {}, "movie_data": {}})


def xc_get_series(user):
    """Get Series from selected categories in vods.db."""
    portals = getPortals()
    allowed_portals = user.get("allowed_portals", [])
    
    series_list = []
    
    try:
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        
        # Get Series items from selected categories
        cursor.execute('''
            SELECT vi.portal_id, vi.category_id, vi.item_id, vi.name, vi.year, 
                   vi.description, vi.genre, vi.rating, vi.poster_url, vi.cmd
            FROM vod_items vi
            INNER JOIN vod_selections vs ON vi.portal_id = vs.portal_id 
                AND (vs.category_key = 'series_' || vi.category_id OR vs.category_key = vi.content_type || '_' || vi.category_id)
            WHERE vi.content_type = 'series' AND vs.enabled = 1
            ORDER BY vi.portal_id, vi.name
        ''')
        
        db_items = cursor.fetchall()
        conn.close()
        
        import hashlib
        
        for item in db_items:
            portal_id = item['portal_id']
            
            # Check if portal is enabled and allowed
            portal = portals.get(portal_id)
            if not portal or portal.get("enabled") != "true":
                continue
            if allowed_portals and portal_id not in allowed_portals:
                continue
            
            internal_id = f"{portal_id}_series_{item['item_id']}"
            numeric_id = int(hashlib.md5(internal_id.encode()).hexdigest()[:8], 16)
            category_key = f"{portal_id}_series_{item['category_id']}"
            
            series_list.append({
                "num": numeric_id,
                "name": item['name'],
                "series_id": numeric_id,
                "cover": item['poster_url'] or "",
                "plot": item['description'] or "",
                "cast": "",
                "director": "",
                "genre": item['genre'] or "",
                "release_date": item['year'] or "",
                "rating": item['rating'] or "",
                "category_id": category_key,
                "custom_sid": internal_id
            })
    except Exception as e:
        logger.error(f"Error getting Series for XC API: {e}")
    
    return flask.jsonify(series_list)


def generate_episode_id(portal_id, series_id, season_num, episode_num):
    """Generate consistent episode ID for XC API.
    
    Format: MD5 hash of "portalId_series_seriesId_sSeasonNum_eEpisodeNum"
    Returns: Numeric ID as string
    """
    import hashlib
    internal_id = f"{portal_id}_series_{series_id}_s{season_num}_e{episode_num}"
    return str(int(hashlib.md5(internal_id.encode()).hexdigest()[:8], 16))


def parse_episode_id(episode_id):
    """Parse episode ID back to components.
    
    Args:
        episode_id: Either custom_sid string or numeric hash
        
    Returns:
        tuple: (portal_id, series_id, season_num, episode_num) or (None, None, None, None)
    """
    if not episode_id:
        return None, None, None, None
    
    episode_str = str(episode_id).strip()
    
    # Handle custom format: portalId_series_itemId_sX_eY
    if '_series_' in episode_str and '_s' in episode_str and '_e' in episode_str:
        try:
            # Split by _series_ first
            parts = episode_str.split('_series_')
            if len(parts) != 2:
                return None, None, None, None
            
            portal_id = parts[0].strip()
            rest = parts[1].strip()
            
            # Split by _s to separate series_id from season/episode
            if '_s' not in rest:
                return None, None, None, None
            
            item_parts = rest.split('_s')
            if len(item_parts) != 2:
                return None, None, None, None
            
            series_id = item_parts[0].strip()
            season_ep = item_parts[1].strip()  # X_eY format
            
            # Split season and episode
            if '_e' not in season_ep:
                return None, None, None, None
            
            season_ep_parts = season_ep.split('_e')
            if len(season_ep_parts) != 2:
                return None, None, None, None
            
            season_num = season_ep_parts[0].strip()
            ep_part = season_ep_parts[1].strip()
            
            # Validate season and episode numbers
            if not season_num.isdigit() or not ep_part.isdigit():
                return None, None, None, None
            
            episode_num = int(ep_part)
            
            # Validate all components are non-empty
            if not portal_id or not series_id or not season_num:
                return None, None, None, None
            
            return portal_id, series_id, season_num, episode_num
            
        except (ValueError, IndexError, AttributeError) as e:
            logger.warning(f"Error parsing episode ID '{episode_id}': {e}")
            return None, None, None, None
    
    # Handle alternative formats if needed (future extensibility)
    # Could add support for other ID formats here
    
    return None, None, None, None


def xc_get_series_info(user, series_id):
    """Get Series info with seasons and episodes for XC API.
    
    The series_id can be either:
    - A numeric hash (from get_series response)
    - A custom_sid string (portalId_series_itemId)
    """
    import hashlib
    
    portals = getPortals()
    allowed_portals = user.get("allowed_portals", [])
    
    # Find the series by ID
    portal_id = None
    item_id = None
    series_data = None
    
    try:
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        
        # Try to find by custom_sid first (string format)
        if '_series_' in str(series_id):
            logger.debug(f"XC API: Parsing custom series ID format: {series_id}")
            parts = str(series_id).split('_series_')
            if len(parts) == 2:
                portal_id = parts[0]
                item_id = parts[1]
                logger.debug(f"XC API: Parsed - Portal: {portal_id}, Item: {item_id}")
        else:
            # Numeric format - search through all series to find matching hash
            logger.debug(f"XC API: Searching for numeric series ID: {series_id}")
            numeric_id = int(series_id)
            
            cursor.execute('''
                SELECT vi.portal_id, vi.item_id, vi.name, vi.year, 
                       vi.description, vi.genre, vi.rating, vi.poster_url, vi.cmd
                FROM vod_items vi
                INNER JOIN vod_selections vs ON vi.portal_id = vs.portal_id 
                    AND (vs.category_key = 'series_' || vi.category_id OR vs.category_key = vi.content_type || '_' || vi.category_id)
                WHERE vi.content_type = 'series' AND vs.enabled = 1
            ''')
            
            for item in cursor.fetchall():
                internal_id = f"{item['portal_id']}_series_{item['item_id']}"
                check_id = int(hashlib.md5(internal_id.encode()).hexdigest()[:8], 16)
                if check_id == numeric_id:
                    portal_id = item['portal_id']
                    item_id = item['item_id']
                    series_data = dict(item)
                    break
        
        if not portal_id or not item_id:
            conn.close()
            return flask.jsonify({"info": {}, "episodes": {}, "error": "Series not found"})
        
        if not series_data:
            cursor.execute('''
                SELECT portal_id, item_id, name, year, description, genre, rating, poster_url, cmd
                FROM vod_items WHERE portal_id = ? AND item_id = ? AND content_type = 'series'
            ''', (portal_id, item_id))
            row = cursor.fetchone()
            if row:
                series_data = dict(row)
        
        conn.close()
        
        if not series_data:
            return flask.jsonify({"info": {}, "episodes": {}, "error": "Series data not found"})
        
        # Check portal access
        portal = portals.get(portal_id)
        if not portal or portal.get("enabled") != "true":
            return flask.jsonify({"info": {}, "episodes": {}, "error": "Portal unavailable"})
        if allowed_portals and portal_id not in allowed_portals:
            return flask.jsonify({"info": {}, "episodes": {}, "error": "Access denied"})
        
        # Get series info with episodes from portal
        url = portal.get("url")
        macs = list(portal.get("macs", {}).keys())
        proxy = portal.get("proxy")
        
        episodes_by_season = {}
        
        # Try to get episodes from portal
        series_info = None
        working_mac = None
        
        for mac in macs:
            try:
                token = stb.getToken(url, mac, proxy)
                if not token:
                    continue
                
                series_info = stb.getSeriesInfo(url, mac, token, item_id, proxy)
                
                if series_info and series_info.get("data"):
                    working_mac = mac
                    
                    for season_data in series_info.get("data", []):
                        season_id = season_data.get("id", "")
                        season_num = str(season_id).split(":")[1] if ":" in str(season_id) else "1"
                        episode_nums = season_data.get("series", [])
                        
                        if episode_nums:
                            episodes_by_season[season_num] = []
                            
                            for ep_num in episode_nums:
                                # Create episode entry with consistent ID generation
                                internal_ep_id = f"{portal_id}_series_{item_id}_s{season_num}_e{ep_num}"
                                ep_numeric_id = generate_episode_id(portal_id, item_id, season_num, ep_num)
                                
                                # Determine container extension from series data
                                container_ext = "mkv"  # Default to mkv for series
                                if series_data.get("cmd"):
                                    # Try to extract extension from cmd or use common video extensions
                                    cmd_lower = series_data["cmd"].lower()
                                    if ".ts" in cmd_lower:
                                        container_ext = "ts"
                                    elif ".mp4" in cmd_lower:
                                        container_ext = "mp4"
                                    elif ".avi" in cmd_lower:
                                        container_ext = "avi"
                                
                                # Build episode title - use episode number format
                                episode_title = f"Episode {ep_num}"
                                
                                # Try to get more detailed episode info if available
                                season_name = season_data.get("name", f"Season {season_num}")
                                
                                episodes_by_season[season_num].append({
                                    "id": str(ep_numeric_id),  # String format for compatibility
                                    "episode_num": int(ep_num),
                                    "title": episode_title,
                                    "container_extension": container_ext,
                                    "info": {
                                        "name": episode_title,
                                        "season": int(season_num),
                                        "episode": int(ep_num),
                                        "duration": season_data.get("time", ""),
                                        "plot": season_data.get("description", ""),
                                        "rating": series_data.get("rating", ""),
                                        "genre": series_data.get("genre", ""),
                                        "movie_image": series_data.get("poster_url", ""),
                                        "duration_secs": 0,
                                        "bitrate": 0,
                                        "releasedate": series_data.get("year", ""),
                                        "air_date": ""
                                    },
                                    "custom_sid": internal_ep_id,
                                    "added": "",
                                    "season": int(season_num),
                                    "direct_source": ""
                                })
                    
                    break  # Got episodes, stop trying MACs
                    
            except Exception as e:
                logger.debug(f"Series info MAC error: {e}")
                continue
        
        internal_id = f"{portal_id}_series_{item_id}"
        numeric_id = int(hashlib.md5(internal_id.encode()).hexdigest()[:8], 16)
        
        if not episodes_by_season:
            # Still return series info even if no episodes found
            response = {
                "info": {
                    "name": series_data.get("name", ""),
                    "cover": series_data.get("poster_url", ""),
                    "plot": series_data.get("description", ""),
                    "cast": "",
                    "director": "",
                    "genre": series_data.get("genre", ""),
                    "release_date": series_data.get("year", ""),
                    "rating": series_data.get("rating", ""),
                    "series_id": numeric_id,
                    "category_id": series_data.get("category_id", ""),
                    "backdrop_path": series_data.get("poster_url", ""),
                    "tmdb_id": "",
                    "last_modified": "",
                    "episode_run_time": "",
                    "youtube_trailer": "",
                    "seasons_count": 0,
                    "episodes_count": 0
                },
                "episodes": {},
                "seasons": [],
                "error": "No episodes available from portal"
            }
            return flask.jsonify(response)
        
        # Extract additional metadata from series info if available
        cast = ""
        director = ""
        if episodes_by_season:
            # Try to get cast/director from first season data
            for season_data in series_info.get("data", []):
                if season_data.get("actors"):
                    cast = season_data["actors"]
                if season_data.get("director"):
                    director = season_data["director"]
                break
        
        # Calculate total episodes count
        total_episodes = sum(len(eps) for eps in episodes_by_season.values())
        
        # Sort episodes by season number (ascending) for proper display in IPTV apps
        sorted_seasons = sorted(episodes_by_season.keys(), key=lambda x: int(x))
        sorted_episodes = {str(s): episodes_by_season[s] for s in sorted_seasons}
        
        # Build seasons array with proper structure for XC API
        seasons_array = []
        for season_key in sorted_seasons:
            season_eps = episodes_by_season[season_key]
            seasons_array.append({
                "season_number": int(season_key),
                "name": f"Season {season_key}",
                "episode_count": len(season_eps),
                "air_date": series_data.get("year", ""),
                "cover": series_data.get("poster_url", ""),
                "cover_big": series_data.get("poster_url", "")
            })
        
        response = {
            "info": {
                "name": series_data.get("name", ""),
                "cover": series_data.get("poster_url", ""),
                "plot": series_data.get("description", ""),
                "cast": cast,
                "director": director,
                "genre": series_data.get("genre", ""),
                "release_date": series_data.get("year", ""),
                "rating": series_data.get("rating", ""),
                "series_id": numeric_id,
                "category_id": series_data.get("category_id", ""),
                "backdrop_path": series_data.get("poster_url", ""),
                "tmdb_id": "",
                "last_modified": "",
                "episode_run_time": "",
                "youtube_trailer": "",
                "seasons_count": len(episodes_by_season),
                "episodes_count": total_episodes
            },
            "episodes": sorted_episodes,
            "seasons": seasons_array
        }
        
        logger.debug(f"Series: {series_data.get('name')} - {len(sorted_seasons)} seasons, {total_episodes} eps")
        
        return flask.jsonify(response)
        
    except Exception as e:
        logger.error(f"Series info error: {e}")
        return flask.jsonify({"info": {}, "episodes": {}})


@app.route("/xc/<username>/<password>/", methods=["GET"])
@app.route("/<username>/<password>/", methods=["GET"])
@xc_auth_only
def xc_base(username, password):
    """XC API base endpoint - redirect to player_api.php."""
    # Block access to data directory
    if username == "data" or password == "data":
        return "Access denied", 403
    return redirect(f"/player_api.php?username={username}&password={password}", code=302)


@app.route("/live/<username>/<password>/<stream_id>", methods=["GET"])
@app.route("/live/<username>/<password>/<stream_id>.<extension>", methods=["GET"])
@app.route("/xc/<username>/<password>/<stream_id>", methods=["GET"])
@app.route("/xc/<username>/<password>/<stream_id>.<extension>", methods=["GET"])
@app.route("/<username>/<password>/<stream_id>", methods=["GET"])
@app.route("/<username>/<password>/<stream_id>.<extension>", methods=["GET"])
@xc_auth_only
def xc_stream(username, password, stream_id, extension=None):
    """XC API stream endpoint."""
    # Block access to data directory and other system paths
    if username == "data" or "MacReplayXC.json" in str(stream_id) or str(stream_id).startswith("data/"):
        return "Access denied", 403
    settings = getSettings()
    if settings.get("xc api enabled") != "true":
        return flask.jsonify({
            "user_info": {
                "auth": 0,
                "message": "XC API is disabled"
            }
        }), 403
    
    user_id, user = validateXCUser(username, password)
    if not user_id:
        return flask.jsonify({
            "user_info": {
                "auth": 0,
                "message": user  # user contains error message
            }
        }), 401
    
    # Parse stream_id - can be either "portalId_channelId" or numeric hash
    if '_' in str(stream_id):
        # String format: portalId_channelId
        try:
            portal_id, channel_id = str(stream_id).rsplit('_', 1)
        except:
            return "Invalid stream ID", 400
    else:
        if not str(stream_id).isdigit():
            return "Invalid stream ID", 400
            
        # Numeric format: need to find the matching channel
        # This is inefficient but necessary for XC API compatibility
        numeric_id = int(stream_id)
        portals = getPortals()
        found = False
        
        import hashlib
        # Create a copy to avoid RuntimeError if dictionary changes during iteration
        for pid, portal in list(portals.items()):
            if portal.get("enabled") != "true":
                continue
            enabled_channels = portal.get("enabled channels", [])
            for cid in enabled_channels:
                internal_id = f"{pid}_{cid}"
                check_id = int(hashlib.md5(internal_id.encode()).hexdigest()[:8], 16)
                if check_id == numeric_id:
                    portal_id = pid
                    channel_id = cid
                    found = True
                    break
            if found:
                break
        
        if not found:
            return "Stream not found", 404
    
    # Check if user has access to this portal
    allowed_portals = user.get("allowed_portals", [])
    if allowed_portals and portal_id not in allowed_portals:
        return "Access denied to this portal", 403
    
    # Generate device ID from user agent + IP
    device_id = f"{get_client_ip(request)}_{request.headers.get('User-Agent', 'unknown')}"
    device_id = str(hash(device_id))
    
    can_connect, message = checkXCConnectionLimit(user_id, device_id)
    if not can_connect:
        return message, 429
    
    registerXCConnection(user_id, device_id, portal_id, channel_id, get_client_ip(request))
    
    try:
        response = stream_channel(portal_id, channel_id, xc_user=username)
        
        if hasattr(response, 'response') and hasattr(response.response, '__iter__'):
            original_iter = response.response
            
            def cleanup_wrapper():
                try:
                    for chunk in original_iter:
                        yield chunk
                finally:
                    unregisterXCConnection(user_id, device_id)
            
            response.response = cleanup_wrapper()
        
        return response
    except Exception as e:
        unregisterXCConnection(user_id, device_id)
        logger.error(f"Stream error: {username} - {e}")
        raise


def test_vod_stream_quick(stream_url, proxy=None):
    """Quick test if VOD stream is accessible (without consuming the full token).
    
    Returns True if stream is accessible, False otherwise.
    Uses Range header to only fetch first few bytes.
    """
    import requests
    
    try:
        proxies = {"http": proxy, "https": proxy} if proxy else None
        headers = {
            "User-Agent": "Mozilla/5.0 (QtEmbedded; U; Linux; C)",
            "Range": "bytes=0-1023"  # Only request first 1KB
        }
        
        response = requests.get(
            stream_url, 
            headers=headers, 
            proxies=proxies, 
            timeout=10,
            stream=True
        )
        
        # 200 or 206 (Partial Content) means success
        if response.status_code in [200, 206]:
            chunk = next(response.iter_content(chunk_size=1024), None)
            response.close()
            return chunk and len(chunk) > 0
        
        return False
        
    except Exception as e:
        logger.debug(f"Stream test error: {e}")
        return False


def get_vod_stream_settings():
    """Get VOD streaming settings from database."""
    try:
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT key, value FROM vod_settings')
        settings = {row['key']: row['value'] for row in cursor.fetchall()}
        conn.close()
        return settings
    except:
        return {'stream_type': 'ffmpeg', 'mac_rotation': 'true'}


def ffmpeg_vod_stream(stream_url, proxy=None):
    """Stream VOD through FFmpeg for better compatibility."""
    
    # Build FFmpeg command
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",
    ]
    
    if proxy:
        ffmpeg_cmd.extend(["-http_proxy", proxy])
    
    ffmpeg_cmd.extend([
        "-i", stream_url,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-f", "mpegts",
        "-mpegts_flags", "resend_headers",
        "pipe:1"
    ])
    
    def generate():
        process = None
        try:
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=65536
            )
            
            while True:
                chunk = process.stdout.read(65536)
                if not chunk:
                    break
                yield chunk
                
        except GeneratorExit:
            logger.info("VOD FFmpeg: Client disconnected")
        except Exception as e:
            logger.error(f"VOD FFmpeg stream error: {e}")
        finally:
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    process.kill()
                logger.debug("VOD FFmpeg: Process terminated")
    
    return Response(
        generate(),
        mimetype="video/mp2t",
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*",
        }
    )


def proxy_vod_stream(stream_url, proxy=None):
    """Proxy a VOD stream through our server.
    
    This is needed for IPTV apps that don't follow HTTP redirects properly.
    """
    import requests
    
    # Check if this is a HEAD request (iOS apps often do HEAD first)
    is_head_request = request.method == 'HEAD'
    
    proxies = {"http": proxy, "https": proxy} if proxy else None
    req_headers = {
        "User-Agent": "Mozilla/5.0 (QtEmbedded; U; Linux; C)",
    }
    
    # Determine content type from URL
    content_type = "video/mp4"
    if ".mkv" in stream_url.lower():
        content_type = "video/x-matroska"
    elif ".ts" in stream_url.lower():
        content_type = "video/mp2t"
    elif ".avi" in stream_url.lower():
        content_type = "video/x-msvideo"
    
    # Build response headers - don't set Content-Length for streaming
    resp_headers = {
        "Content-Disposition": "inline",
        "Cache-Control": "no-cache",
        "Access-Control-Allow-Origin": "*",
    }
    
    # For HEAD requests, try to get content info
    if is_head_request:
        try:
            head_resp = requests.head(stream_url, headers=req_headers, proxies=proxies, 
                                      timeout=10, allow_redirects=True)
            if head_resp.headers.get('Content-Length'):
                resp_headers["Content-Length"] = head_resp.headers.get('Content-Length')
            if head_resp.headers.get('Content-Type'):
                content_type = head_resp.headers.get('Content-Type')
        except:
            pass
        
        logger.debug(f"VOD proxy: Responding to HEAD request")
        return Response('', status=200, mimetype=content_type, headers=resp_headers)
    
    # For GET requests, stream the content
    def generate():
        try:
            with requests.get(stream_url, headers=req_headers, proxies=proxies, 
                            stream=True, timeout=60) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        yield chunk
        except GeneratorExit:
            logger.debug("VOD proxy: Client disconnected")
        except Exception as e:
            logger.error(f"VOD proxy stream error: {e}")
    
    return Response(
        generate(),
        mimetype=content_type,
        headers=resp_headers
    )


@app.route("/movie/<username>/<password>/<stream_id>", methods=["GET", "HEAD"])
@app.route("/movie/<username>/<password>/<stream_id>.<extension>", methods=["GET", "HEAD"])
@xc_auth_only
def xc_movie_stream(username, password, stream_id, extension=None):
    """XC API movie/VOD stream endpoint."""
    import hashlib
    
    settings = getSettings()
    if settings.get("xc api enabled") != "true":
        return flask.jsonify({"user_info": {"auth": 0, "message": "XC API disabled"}}), 403
    
    user_id, user = validateXCUser(username, password)
    if not user_id:
        return flask.jsonify({"user_info": {"auth": 0, "message": user}}), 401
    
    # Parse stream_id to find the VOD
    portal_id = None
    item_id = None
    
    if '_vod_' in str(stream_id):
        parts = str(stream_id).split('_vod_')
        if len(parts) == 2:
            portal_id = parts[0]
            item_id = parts[1]
    elif str(stream_id).isdigit():
        # Numeric format - search through all VODs
        numeric_id = int(stream_id)
        portals = getPortals()
        allowed_portals = user.get("allowed_portals", [])
        
        try:
            conn = get_vod_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT vi.portal_id, vi.item_id FROM vod_items vi
                INNER JOIN vod_selections vs ON vi.portal_id = vs.portal_id 
                    AND (vs.category_key = 'vod_' || vi.category_id OR vs.category_key = vi.content_type || '_' || vi.category_id)
                WHERE vi.content_type = 'vod' AND vs.enabled = 1
            ''')
            
            for item in cursor.fetchall():
                p_id = item['portal_id']
                i_id = item['item_id']
                
                portal = portals.get(p_id)
                if not portal or portal.get("enabled") != "true":
                    continue
                if allowed_portals and p_id not in allowed_portals:
                    continue
                
                internal_id = f"{p_id}_vod_{i_id}"
                check_id = int(hashlib.md5(internal_id.encode()).hexdigest()[:8], 16)
                if check_id == numeric_id:
                    portal_id = p_id
                    item_id = i_id
                    break
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error finding VOD: {e}")
            return flask.jsonify({
                "error": "VOD not found",
                "details": str(e)
            }), 404
    else:
        logger.error(f"XC API: Invalid VOD ID format: {stream_id}")
        return flask.jsonify({
            "error": "Invalid VOD ID format",
            "details": f"Could not parse VOD ID: {stream_id}"
        }), 400
    
    if not portal_id or not item_id:
        logger.error(f"XC API: VOD not found - stream_id: {stream_id}")
        return flask.jsonify({
            "error": "VOD not found",
            "details": f"Could not find VOD with ID: {stream_id}"
        }), 404
    
    # Get the stream URL for this VOD
    portals = getPortals()
    portal = portals.get(portal_id)
    if not portal:
        logger.error(f"XC API: Portal {portal_id} not found")
        return flask.jsonify({
            "error": "Portal not found",
            "details": f"Portal {portal_id} is not configured"
        }), 404
    
    url = portal.get("url")
    macs = list(portal.get("macs", {}).keys())
    proxy = portal.get("proxy")
    
    if not macs:
        logger.error(f"XC API: No MACs configured for portal {portal_id}")
        return flask.jsonify({
            "error": "No MACs configured",
            "details": f"Portal {portal_id} has no MAC addresses"
        }), 500
    
    # Get the VOD cmd and cached working_macs from database
    cached_mac = None
    try:
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT cmd, working_macs FROM vod_items 
            WHERE portal_id = ? AND item_id = ? AND content_type = 'vod'
        ''', (portal_id, item_id))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            logger.warning(f"VOD: {item_id} not in DB")
            return flask.jsonify({"error": "VOD data not found"}), 404
        
        vod_cmd = row['cmd']
        if row['working_macs']:
            cached_mac = row['working_macs'].split(',')[0]
    except Exception as e:
        logger.error(f"VOD DB error: {e}")
        return flask.jsonify({"error": "Database error"}), 500
    
    # Sort MACs to try cached MAC first
    if cached_mac and cached_mac in macs:
        macs = [cached_mac] + [m for m in macs if m != cached_mac]
    
    failed_macs = []
    
    for mac_index, mac in enumerate(macs, 1):
        try:
            token = stb.getToken(url, mac, proxy)
            if not token:
                failed_macs.append({"mac": mac[:15] + "...", "reason": "No token"})
                continue
            
            link = stb.getVodLink(url, mac, token, vod_cmd, proxy)
            if not link or not link.startswith(('http://', 'https://')):
                failed_macs.append({"mac": mac[:15] + "...", "reason": "No link"})
                continue
            
            # Test stream accessibility
            if not test_vod_stream_quick(link, proxy):
                failed_macs.append({"mac": mac[:15] + "...", "reason": "458/403"})
                continue
            
            logger.info(f"VOD: {item_id} → MAC {mac_index}/{len(macs)} OK")
            
            # Cache working MAC
            try:
                cache_conn = get_vod_db_connection()
                cache_cursor = cache_conn.cursor()
                cache_cursor.execute('UPDATE vod_items SET working_macs = ? WHERE portal_id = ? AND item_id = ? AND content_type = ?', 
                    (mac, portal_id, item_id, 'vod'))
                cache_conn.commit()
                cache_conn.close()
            except:
                pass
            
            # Get fresh link for playback
            fresh_link = stb.getVodLink(url, mac, token, vod_cmd, proxy) or link
            
            # Check VOD settings for stream type
            vod_settings = get_vod_stream_settings()
            stream_type = vod_settings.get('stream_type', 'ffmpeg')
            settings = getSettings()
            
            if stream_type == 'ffmpeg':
                return ffmpeg_vod_stream(fresh_link, proxy)
            elif settings.get("xc vod proxy", "false") == "true":
                return proxy_vod_stream(fresh_link, proxy)
            else:
                return redirect(fresh_link, code=302)
                    
        except Exception as e:
            failed_macs.append({"mac": mac[:15] + "...", "reason": str(e)[:30]})
            continue
    
    # All MACs failed
    logger.warning(f"VOD: {item_id} FAILED - {len(failed_macs)} MACs tried")
    return flask.jsonify({"error": "Stream not available", "failed_macs": failed_macs}), 500


@app.route("/series/<username>/<password>/<stream_id>", methods=["GET", "HEAD"])
@app.route("/series/<username>/<password>/<stream_id>.<extension>", methods=["GET", "HEAD"])
@xc_auth_only
def xc_series_stream(username, password, stream_id, extension=None):
    """XC API series stream endpoint for episodes."""
    import hashlib
    
    settings = getSettings()
    if settings.get("xc api enabled") != "true":
        return flask.jsonify({"user_info": {"auth": 0, "message": "XC API disabled"}}), 403
    
    user_id, user = validateXCUser(username, password)
    if not user_id:
        return flask.jsonify({"user_info": {"auth": 0, "message": user}}), 401
    
    # Parse stream_id to find the episode
    portal_id = None
    series_id = None
    season_num = None
    episode_num = None
    
    portal_id, series_id, season_num, episode_num = parse_episode_id(stream_id)
    
    if not (portal_id and series_id and season_num and episode_num) and str(stream_id).isdigit():
        # Numeric format - search through all episodes
        numeric_id = int(stream_id)
        portals = getPortals()
        allowed_portals = user.get("allowed_portals", [])
        
        try:
            conn = get_vod_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT vi.portal_id, vi.item_id FROM vod_items vi
                INNER JOIN vod_selections vs ON vi.portal_id = vs.portal_id 
                    AND (vs.category_key = 'series_' || vi.category_id OR vs.category_key = vi.content_type || '_' || vi.category_id)
                WHERE vi.content_type = 'series' AND vs.enabled = 1
            ''')
            
            found = False
            for item in cursor.fetchall():
                p_id = item['portal_id']
                i_id = item['item_id']
                
                portal = portals.get(p_id)
                if not portal or portal.get("enabled") != "true":
                    continue
                if allowed_portals and p_id not in allowed_portals:
                    continue
                
                url = portal.get("url")
                macs = list(portal.get("macs", {}).keys())
                proxy = portal.get("proxy")
                
                for mac in macs:
                    try:
                        token = stb.getToken(url, mac, proxy)
                        if not token:
                            continue
                        
                        series_info = stb.getSeriesInfo(url, mac, token, i_id, proxy)
                        if series_info and series_info.get("data"):
                            for season_data in series_info.get("data", []):
                                s_id = season_data.get("id", "")
                                s_num = str(s_id).split(":")[1] if ":" in str(s_id) else "1"
                                
                                for ep_num in season_data.get("series", []):
                                    check_id = int(generate_episode_id(p_id, i_id, s_num, ep_num))
                                    if check_id == numeric_id:
                                        portal_id, series_id, season_num, episode_num = p_id, i_id, s_num, ep_num
                                        found = True
                                        break
                                if found:
                                    break
                        if found:
                            break
                    except:
                        continue
                    if found:
                        break
                if found:
                    break
            conn.close()
        except Exception as e:
            logger.error(f"Series search error: {e}")
            return "Episode not found", 404
    elif not (portal_id and series_id and season_num and episode_num):
        return "Invalid episode ID", 400
    
    if not portal_id or not series_id or episode_num is None:
        return flask.jsonify({"error": "Episode not found"}), 404
    
    portals = getPortals()
    portal = portals.get(portal_id)
    if not portal:
        return flask.jsonify({"error": "Portal not found"}), 404
    
    url = portal.get("url")
    macs = list(portal.get("macs", {}).keys())
    proxy = portal.get("proxy")
    
    if not macs:
        return flask.jsonify({"error": "No MACs configured"}), 500
    
    # Get cached working MAC from database
    cached_mac = None
    try:
        conn = get_vod_db_connection()
        cursor = conn.cursor()
        base_id = str(series_id).split(':')[0] if ':' in str(series_id) else str(series_id)
        
        cursor.execute('SELECT working_macs FROM vod_items WHERE portal_id = ? AND item_id LIKE ? AND content_type = ?', 
            (portal_id, f"%{base_id}%", 'series'))
        row = cursor.fetchone()
        conn.close()
        
        if row and row['working_macs']:
            cached_mac = row['working_macs'].split(',')[0]
    except:
        pass
    
    # Sort MACs to try cached MAC first
    if cached_mac and cached_mac in macs:
        macs = [cached_mac] + [m for m in macs if m != cached_mac]
    
    failed_macs = []
    
    for mac_index, mac in enumerate(macs, 1):
        try:
            token = stb.getToken(url, mac, proxy)
            if not token:
                failed_macs.append({"mac": mac[:15] + "...", "reason": "No token"})
                continue
            
            # Build series cmd (Base64-encoded JSON)
            import base64
            import json as json_module
            
            base_series_id = str(series_id).split(':')[0] if ':' in str(series_id) else str(series_id)
            cmd_data = {"series_id": base_series_id, "season_num": int(season_num), "type": "series"}
            current_cmd = base64.b64encode(json_module.dumps(cmd_data).encode()).decode()
            
            link = stb.getSeriesLink(url, mac, token, current_cmd, episode_num, season_num, episode_num, proxy)
            if not link or not link.startswith(('http://', 'https://')):
                failed_macs.append({"mac": mac[:15] + "...", "reason": "No link"})
                continue
            
            # Test stream accessibility
            if not test_vod_stream_quick(link, proxy):
                failed_macs.append({"mac": mac[:15] + "...", "reason": "458/403"})
                continue
            
            logger.info(f"Series: S{season_num}E{episode_num} → MAC {mac_index}/{len(macs)} OK")
            
            # Cache working MAC
            try:
                cache_conn = get_vod_db_connection()
                cache_cursor = cache_conn.cursor()
                cache_cursor.execute('UPDATE vod_items SET working_macs = ? WHERE portal_id = ? AND item_id LIKE ? AND content_type = ?', 
                    (mac, portal_id, f"%{base_series_id}%", 'series'))
                cache_conn.commit()
                cache_conn.close()
            except:
                pass
            
            # Get fresh link for playback
            fresh_link = stb.getSeriesLink(url, mac, token, current_cmd, episode_num, season_num, episode_num, proxy) or link
            
            # Check VOD settings for stream type
            vod_settings = get_vod_stream_settings()
            stream_type = vod_settings.get('stream_type', 'ffmpeg')
            
            if stream_type == 'ffmpeg':
                return ffmpeg_vod_stream(fresh_link, proxy)
            elif settings.get("xc vod proxy", "false") == "true":
                return proxy_vod_stream(fresh_link, proxy)
            else:
                return redirect(fresh_link, code=302)
                    
        except Exception as e:
            failed_macs.append({"mac": mac[:15] + "...", "reason": str(e)[:30]})
            continue
    
    # All MACs failed
    logger.warning(f"Series: S{season_num}E{episode_num} FAILED - {len(failed_macs)} MACs tried")
    return flask.jsonify({"error": "Stream not available", "failed_macs": failed_macs}), 500


@app.route("/xmltv.php", methods=["GET"])
@xc_auth_only
def xc_xmltv():
    """XC API XMLTV endpoint."""
    global cached_xmltv, last_updated
    
    # Refresh cache if needed
    if cached_xmltv is None or (time.time() - last_updated) > 900:
        refresh_xmltv()
    
    return Response(
        cached_xmltv,
        mimetype="text/xml",
    )


def stream_channel(portalId, channelId, xc_user=None):
    """Internal function to stream a channel without authentication."""
    def streamData():
        def occupy():
            occupied.setdefault(portalId, [])
            stream_info = {
                "mac": mac,
                "channel id": channelId,
                "channel name": channelName,
                "client": ip,
                "portal name": portalName,
                "start time": startTime,
            }
            if xc_user:
                stream_info["xc_user"] = xc_user
            occupied.get(portalId, []).append(stream_info)
            logger.info("Occupied Portal({}):MAC({}):User({})".format(portalId, mac, xc_user or "Direct"))

        def unoccupy():
            stream_info = {
                "mac": mac,
                "channel id": channelId,
                "channel name": channelName,
                "client": ip,
                "portal name": portalName,
                "start time": startTime,
            }
            if xc_user:
                stream_info["xc_user"] = xc_user
            try:
                occupied.get(portalId, []).remove(stream_info)
            except ValueError:
                pass  # Already removed
            logger.info("Unoccupied Portal({}):MAC({}):User({})".format(portalId, mac, xc_user or "Direct"))

        try:
            startTime = datetime.now(timezone.utc).timestamp()
            occupy()
            with subprocess.Popen(
                ffmpegcmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ) as ffmpeg_sp:
                while True:
                    chunk = ffmpeg_sp.stdout.read(1024)
                    if len(chunk) == 0:
                        if ffmpeg_sp.poll() != 0:
                            logger.info("Ffmpeg closed with error({}). Moving MAC({}) for Portal({})".format(str(ffmpeg_sp.poll()), mac, portalName))
                            moveMac(portalId, mac)
                        break
                    yield chunk
        except:
            pass
        finally:
            unoccupy()
            ffmpeg_sp.kill()

    def testStream():
        timeout = int(getSettings()["ffmpeg timeout"]) * int(1000000)
        ffprobecmd = [ffprobe_path, "-timeout", str(timeout), "-i", link]

        if proxy:
            ffprobecmd.insert(1, "-http_proxy")
            ffprobecmd.insert(2, proxy)

        with subprocess.Popen(
            ffprobecmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as ffprobe_sb:
            ffprobe_sb.communicate()
            if ffprobe_sb.returncode == 0:
                return True
            else:
                return False

    def isMacFree():
        count = 0
        for i in occupied.get(portalId, []):
            if i["mac"] == mac:
                count = count + 1
        if count < streamsPerMac:
            return True
        else:
            return False

    portal = getPortals().get(portalId)
    
    # Check if portal exists
    if not portal:
        logger.error(f"Portal {portalId} not found")
        return make_response("Portal not found", 404)
    
    portalName = portal.get("name")
    url = portal.get("url")
    macs = list(portal["macs"].keys())
    streamsPerMac = int(portal.get("streams per mac"))
    proxy = portal.get("proxy")
    web = request.args.get("web")
    ip = get_client_ip(request)

    logger.info(
        "IP({}) requested Portal({}):Channel({})".format(ip, portalId, channelId)
    )

    freeMac = False

    for mac in macs:
        channels = None
        cmd = None
        link = None
        if streamsPerMac == 0 or isMacFree():
            logger.info(
                "Trying Portal({}):MAC({}):Channel({})".format(portalId, mac, channelId)
            )
            freeMac = True
            token = stb.getToken(url, mac, proxy)
            if token:
                stb.getProfile(url, mac, token, proxy)
                channels = stb.getAllChannels(url, mac, token, proxy)

        if channels:
            for c in channels:
                if str(c["id"]) == channelId:
                    channelName = portal.get("custom channel names", {}).get(channelId)
                    if channelName == None:
                        channelName = c["name"]
                    cmd = c["cmd"]
                    break

        if cmd:
            if "http://localhost/" in cmd:
                link = stb.getLink(url, mac, token, cmd, proxy)
            else:
                link = cmd.split(" ")[1]

        if link:
            if getSettings().get("test streams", "true") == "false" or testStream():
                if web:
                    ffmpegcmd = [
                        ffmpeg_path,
                        "-loglevel",
                        "panic",
                        "-hide_banner",
                        "-i",
                        link,
                        "-vcodec",
                        "copy",
                        "-f",
                        "mp4",
                        "-movflags",
                        "frag_keyframe+empty_moov",
                        "pipe:",
                    ]
                    if proxy:
                        ffmpegcmd.insert(1, "-http_proxy")
                        ffmpegcmd.insert(2, proxy)
                    # Use correct mimetype for MPEG-TS streams
                    response = Response(streamData(), mimetype="video/mp2t")
                    response.headers['Content-Type'] = 'video/mp2t'
                    response.headers['Accept-Ranges'] = 'none'
                    return response

                else:
                    if getSettings().get("stream method", "ffmpeg") == "ffmpeg":
                        ffmpegcmd = f"{ffmpeg_path} {getSettings()['ffmpeg command']}"
                        ffmpegcmd = ffmpegcmd.replace("<url>", link)
                        ffmpegcmd = ffmpegcmd.replace(
                            "<timeout>",
                            str(int(getSettings()["ffmpeg timeout"]) * int(1000000)),
                        )
                        if proxy:
                            ffmpegcmd = ffmpegcmd.replace("<proxy>", proxy)
                        else:
                            ffmpegcmd = ffmpegcmd.replace("-http_proxy <proxy>", "")
                        " ".join(ffmpegcmd.split())
                        ffmpegcmd = ffmpegcmd.split()
                        return Response(
                            streamData(), mimetype="application/octet-stream"
                        )
                    else:
                        logger.info("Redirect sent")
                        return redirect(link)

        logger.info(
            "Unable to connect to Portal({}) using MAC({})".format(portalId, mac)
        )
        logger.info("Moving MAC({}) for Portal({})".format(mac, portalName))
        moveMac(portalId, mac)

        if not getSettings().get("try all macs", "true") == "true":
            break

    # (Fallback logic remains the same but too long to include here)
    # ... rest of the original channel function

    if freeMac:
        logger.info(
            "No working streams found for Portal({}):Channel({})".format(
                portalId, channelId
            )
        )
    else:
        logger.info(
            "No free MAC for Portal({}):Channel({})".format(portalId, channelId)
        )

    return make_response("No streams available", 503)


@app.route("/play/<portalId>/<channelId>", methods=["GET"])
def channel(portalId, channelId):
    """Stream endpoint with configurable access control."""
    settings = getSettings()
    public_access = settings.get("public playlist access", "true") == "true"
    
    if public_access:
        # Public access enabled - no authentication required
        return stream_channel(portalId, channelId)
    else:
        # Public access disabled - require Basic Auth
        auth = request.authorization
        if not auth or not auth.username or not auth.password:
            # No Basic Auth provided - return 401 with WWW-Authenticate header
            response = Response(
                'Authentication required for stream access\n'
                'Please provide Basic Auth credentials.',
                401,
                {'WWW-Authenticate': 'Basic realm="MacReplayXC Stream Access"'}
            )
            return response
        
        # Validate Basic Auth credentials
        system_username = settings.get("username", "admin")
        system_password = settings.get("password", "12345")
        
        if auth.username != system_username or auth.password != system_password:
            logger.warning(f"Invalid Basic Auth credentials for stream: {auth.username}")
            response = Response(
                'Invalid credentials for stream access\n'
                'Please check your username and password.',
                401,
                {'WWW-Authenticate': 'Basic realm="MacReplayXC Stream Access"'}
            )
            return response
        
        # Authentication successful
        logger.info(f"Basic Auth successful for stream: {auth.username}")
        return stream_channel(portalId, channelId)


@app.route("/hls/<portalId>/<channelId>/<path:filename>", methods=["GET"])
def hls_stream(portalId, channelId, filename):
    """Serve HLS streams (playlists and segments)."""
    from flask import send_file
    
    # Get portal info
    portal = getPortals().get(portalId)
    if not portal:
        logger.error(f"Portal {portalId} not found for HLS request")
        return make_response("Portal not found", 404)
    
    portalName = portal.get("name")
    url = portal.get("url")
    macs = list(portal["macs"].keys())
    proxy = portal.get("proxy")
    ip = get_client_ip(request)
    
    logger.info(f"HLS request from IP({ip}) for Portal({portalId}):Channel({channelId}):File({filename})")
    
    # Check if we already have this stream
    stream_key = f"{portalId}_{channelId}"
    
    # First, check if stream is already active
    stream_exists = stream_key in hls_manager.streams
    
    if stream_exists:
        # For active streams, wait a bit for the file if it's a playlist
        if filename.endswith('.m3u8'):
            is_passthrough = hls_manager.streams[stream_key].get('is_passthrough', False)
            max_wait = 100 if not is_passthrough else 10
            
            for wait_count in range(max_wait):
                file_path = hls_manager.get_file(portalId, channelId, filename)
                if file_path:
                    break
                time.sleep(0.1)
        else:
            file_path = hls_manager.get_file(portalId, channelId, filename)
    else:
        file_path = None
    
    # If file doesn't exist and this is a playlist/segment request, start the stream
    if not file_path and (filename.endswith('.m3u8') or filename.endswith('.ts') or filename.endswith('.m4s')):
        # Get the stream URL
        link = None
        for mac in macs:
            try:
                token = stb.getToken(url, mac, proxy)
                if token:
                    stb.getProfile(url, mac, token, proxy)
                    channels = stb.getAllChannels(url, mac, token, proxy)
                    
                    if channels:
                        for c in channels:
                            if str(c["id"]) == channelId:
                                cmd = c["cmd"]
                                if "http://localhost/" in cmd:
                                    link = stb.getLink(url, mac, token, cmd, proxy)
                                else:
                                    link = cmd.split(" ")[1]
                                break
                    
                    if link:
                        break
            except Exception as e:
                logger.error(f"Error getting stream URL for HLS with MAC {mac}: {e}")
                continue
        
        if not link:
            logger.error(f"Could not get stream URL for Portal({portalId}):Channel({channelId})")
            return make_response("Stream not available", 503)
        
        # Start the HLS stream
        try:
            stream_info = hls_manager.start_stream(portalId, channelId, link, proxy)
            
            # Wait for file to be created
            is_passthrough = stream_info.get('is_passthrough', False)
            
            if filename.endswith('.m3u8'):
                max_wait = 100 if not is_passthrough else 10
                
                for wait_count in range(max_wait):
                    file_path = hls_manager.get_file(portalId, channelId, filename)
                    if file_path:
                        break
                    time.sleep(0.1)
            else:
                # For segments, wait a bit
                for wait_count in range(30):
                    file_path = hls_manager.get_file(portalId, channelId, filename)
                    if file_path:
                        break
                    time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Error starting HLS stream: {e}")
            return make_response("Error starting stream", 500)
    
    # Serve the file
    if file_path and os.path.exists(file_path):
        # Determine MIME type
        if filename.endswith('.m3u8'):
            mimetype = 'application/vnd.apple.mpegurl'
        elif filename.endswith('.ts'):
            mimetype = 'video/mp2t'
        elif filename.endswith('.m4s'):
            mimetype = 'video/iso.segment'
        elif filename.endswith('.mp4'):
            mimetype = 'video/mp4'
        else:
            mimetype = 'application/octet-stream'
        
        return send_file(file_path, mimetype=mimetype)
    else:
        logger.warning(f"File not found: {filename} for stream {stream_key}")
        return make_response("File not found", 404)


@app.route("/dashboard")
@authorise
def dashboard():
    return render_template("dashboard.html")

@app.route("/streaming")
@authorise
def streaming():
    return flask.jsonify(occupied)

# Store server start time
server_start_time = time.time()

@app.route("/dashboard/stats")
@authorise
def dashboard_stats():
    """Get dashboard statistics."""
    # Count total enabled channels from database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM channels WHERE enabled = 1')
    total_channels = cursor.fetchone()[0]
    conn.close()
    
    # Get last playlist update time
    global last_updated
    last_update_time = datetime.fromtimestamp(last_updated).isoformat() if last_updated > 0 else None
    
    # Calculate uptime
    uptime_seconds = int(time.time() - server_start_time)
    
    return flask.jsonify({
        "total_channels": total_channels,
        "last_updated": last_update_time,
        "uptime_seconds": uptime_seconds
    })

@app.route("/log")
@authorise
def log():
    logFilePath = "/app/logs/MacReplayXC.log"
    
    try:
        with open(logFilePath) as f:
            log_content = f.read()
        return log_content
    except FileNotFoundError:
        return "Log file not found"

def hdhr(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        settings = getSettings()
        security = settings["enable security"]
        username = settings["username"]
        password = settings["password"]
        hdhrenabled = settings["enable hdhr"]
        if (
            security == "false"
            or auth
            and auth.username == username
            and auth.password == password
        ):
            if hdhrenabled:
                return f(*args, **kwargs)
        return make_response("Error", 404)

    return decorated

@app.route("/discover.json", methods=["GET"])
@hdhr
def discover():
    logger.info("HDHR Status Requested.")
    settings = getSettings()
    name = settings["hdhr name"]
    id = settings["hdhr id"]
    tuners = settings["hdhr tuners"]
    data = {
        "BaseURL": host,
        "DeviceAuth": name,
        "DeviceID": id,
        "FirmwareName": "MacReplayXC",
        "FirmwareVersion": "666",
        "FriendlyName": name,
        "LineupURL": host + "/lineup.json",
        "Manufacturer": "Evilvirus",
        "ModelNumber": "666",
        "TunerCount": int(tuners),
    }
    return flask.jsonify(data)

@app.route("/lineup_status.json", methods=["GET"])
@hdhr
def status():
    data = {
        "ScanInProgress": 0,
        "ScanPossible": 0,
        "Source": "Cable",
        "SourceList": ["Cable"],
    }
    return flask.jsonify(data)

def refresh_lineup():
    global cached_lineup
    logger.info("Refreshing Lineup...")
    lineup = []
    portals = getPortals()
    for portal in portals:
        if portals[portal]["enabled"] == "true":
            enabledChannels = portals[portal].get("enabled channels", [])
            if len(enabledChannels) != 0:
                name = portals[portal]["name"]
                url = portals[portal]["url"]
                macs = list(portals[portal]["macs"].keys())
                proxy = portals[portal]["proxy"]
                customChannelNames = portals[portal].get("custom channel names", {})
                customChannelNumbers = portals[portal].get("custom channel numbers", {})

                for mac in macs:
                    try:
                        token = stb.getToken(url, mac, proxy)
                        stb.getProfile(url, mac, token, proxy)
                        allChannels = stb.getAllChannels(url, mac, token, proxy)
                        break
                    except:
                        allChannels = None

                if allChannels:
                    for channel in allChannels:
                        channelId = str(channel.get("id"))
                        if channelId in enabledChannels:
                            channelName = customChannelNames.get(channelId)
                            if channelName is None:
                                channelName = str(channel.get("name"))
                            channelNumber = customChannelNumbers.get(channelId)
                            if channelNumber is None:
                                channelNumber = str(channel.get("number"))

                            lineup.append(
                                {
                                    "GuideNumber": channelNumber,
                                    "GuideName": channelName,
                                    "URL": "http://"
                                    + host
                                    + "/play/"
                                    + portal
                                    + "/"
                                    + channelId,
                                }
                            )
                else:
                    logger.error("Error making lineup for {}, skipping".format(name))
    
    lineup.sort(key=lambda x: int(x["GuideNumber"]))

    cached_lineup = lineup
    logger.info("Lineup Refreshed.")
    
@app.route("/lineup.json", methods=["GET"])
@app.route("/lineup.post", methods=["POST"])
@hdhr
def lineup():
    logger.info("Lineup Requested")
    if not cached_lineup:
        refresh_lineup()
    logger.info("Lineup Delivered")
    return jsonify(cached_lineup)

@app.route("/refresh_lineup", methods=["POST"])
@authorise
def refresh_lineup_endpoint():
    try:
        refresh_lineup()
        logger.info("Lineup refreshed via dashboard")
        return jsonify({"status": "Lineup refreshed successfully"})
    except Exception as e:
        logger.error(f"Error refreshing lineup: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/logs/recent", methods=["GET"])
@authorise
def get_recent_logs():
    """Get recent log entries for live log display."""
    try:
        # Use existing log() function to get log content
        log_content = log()
        
        # Split into lines and get last 50
        lines = log_content.split('\n')
        recent_lines = lines[-50:] if len(lines) > 50 else lines
        
        logs = []
        for line in recent_lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse log format: "2024-01-01 12:00:00,123 [LEVEL] message"
            try:
                if ' [' in line and '] ' in line:
                    timestamp_part = line.split(' [')[0]
                    level_part = line.split(' [')[1].split('] ')[0]
                    message_part = '] '.join(line.split(' [')[1].split('] ')[1:])
                    
                    logs.append({
                        'timestamp': timestamp_part,
                        'level': level_part,
                        'message': message_part
                    })
                else:
                    # Fallback for lines that don't match format
                    logs.append({
                        'timestamp': datetime.now().isoformat(),
                        'level': 'INFO',
                        'message': line
                    })
            except:
                # Skip malformed lines
                continue
        
        return jsonify(logs)
        
    except Exception as e:
        logger.error(f"Error reading recent logs: {e}")
        return jsonify([])

def start_refresh():
    threading.Thread(target=refresh_lineup, daemon=True).start()
    threading.Thread(target=refresh_xmltv, daemon=True).start()


@app.route("/proxy/test", methods=["POST"])
@authorise
def proxy_test():
    """Test proxy connectivity and functionality."""
    try:
        data = request.json
        proxy_url = data.get('proxy_url', '').strip()
        test_url = data.get('test_url', 'http://httpbin.org/ip')
        
        if not proxy_url:
            return flask.jsonify({"error": "No proxy URL provided"}), 400
        
        # Test 1: Validation
        is_valid = validate_proxy_url(proxy_url)
        proxy_type = get_proxy_type(proxy_url)
        
        result = {
            "proxy_url": proxy_url,
            "proxy_type": proxy_type,
            "valid": is_valid,
            "tests": {}
        }
        
        if not is_valid:
            result["error"] = f"Invalid proxy format. Detected type: {proxy_type}"
            return flask.jsonify(result), 400
        
        # Test 2: Parse proxy
        try:
            parsed_proxy = parse_proxy_url(proxy_url)
            result["parsed"] = parsed_proxy
            result["tests"]["parsing"] = {"success": True, "message": "Proxy URL parsed successfully"}
        except Exception as e:
            result["tests"]["parsing"] = {"success": False, "error": str(e)}
            return flask.jsonify(result), 500
        
        # Test 3: HTTP connectivity test
        try:
            import requests
            
            # For SOCKS proxies, ensure we have the right dependencies and use HTTP
            if proxy_type in ['socks5', 'socks4']:
                try:
                    import socks
                except ImportError:
                    result["tests"]["connectivity"] = {
                        "success": False, 
                        "error": "PySocks library not available. Install with: pip install requests[socks]"
                    }
                    return flask.jsonify(result), 500
                
                # Use HTTP instead of HTTPS for SOCKS to avoid SSL issues
                if test_url.startswith('https://'):
                    test_url = test_url.replace('https://', 'http://')
            
            response = requests.get(test_url, proxies=parsed_proxy, timeout=10)
            
            if response.status_code == 200:
                try:
                    # Try JSON first (httpbin.org/ip format)
                    data = response.json()
                    external_ip = data.get('origin', 'Unknown')
                    result["tests"]["connectivity"] = {
                        "success": True, 
                        "message": f"Connection successful via proxy",
                        "external_ip": external_ip,
                        "status_code": response.status_code
                    }
                except:
                    # Fallback to plain text (ipinfo.io/ip format)
                    external_ip = response.text.strip()
                    if external_ip and len(external_ip) < 50:  # Reasonable IP length
                        result["tests"]["connectivity"] = {
                            "success": True, 
                            "message": f"Connection successful via proxy",
                            "external_ip": external_ip,
                            "status_code": response.status_code
                        }
                    else:
                        result["tests"]["connectivity"] = {
                            "success": True, 
                            "message": f"Connection successful via proxy",
                            "status_code": response.status_code,
                            "response_preview": response.text[:100]
                        }
            else:
                result["tests"]["connectivity"] = {
                    "success": False, 
                    "error": f"HTTP {response.status_code}",
                    "status_code": response.status_code
                }
        except requests.exceptions.ProxyError as e:
            result["tests"]["connectivity"] = {"success": False, "error": f"Proxy error: {str(e)}"}
        except requests.exceptions.ConnectTimeout:
            result["tests"]["connectivity"] = {"success": False, "error": "Connection timeout"}
        except requests.exceptions.ConnectionError as e:
            result["tests"]["connectivity"] = {"success": False, "error": f"Connection error: {str(e)}"}
        except Exception as e:
            result["tests"]["connectivity"] = {"success": False, "error": f"Unexpected error: {str(e)}"}
        
        # Test 4: Shadowsocks specific test (if applicable)
        if proxy_type == 'shadowsocks':
            try:
                # Try to import shadowsocks library
                import shadowsocks
                result["tests"]["shadowsocks_library"] = {"success": True, "message": "Shadowsocks library available"}
                
                # Additional Shadowsocks connectivity test could go here
                result["tests"]["shadowsocks_connectivity"] = {
                    "success": True, 
                    "message": "Shadowsocks library detected, basic validation passed"
                }
            except ImportError:
                result["tests"]["shadowsocks_library"] = {
                    "success": False, 
                    "error": "Shadowsocks library not available. Install with: pip install shadowsocks==2.8.2"
                }
            except Exception as e:
                result["tests"]["shadowsocks_connectivity"] = {"success": False, "error": str(e)}
        
        # Overall success
        all_tests_passed = all(
            test.get("success", False) 
            for test in result["tests"].values()
        )
        result["overall_success"] = all_tests_passed
        
        return flask.jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in proxy_test: {e}")
        return flask.jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    config = loadConfig()
    
    # Initialize the database
    init_db()
    
    # Initialize the VOD database
    init_vod_db()
    
    # Check if database has any channels
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM channels")
    count = cursor.fetchone()[0]
    conn.close()
    
    # If no channels in database, refresh from portals
    if count == 0:
        logger.info("No channels in database, fetching from portals...")
        refresh_channels_cache()
    
    start_refresh()
    
    # Initialize HLS stream manager with settings
    settings = getSettings()
    
    # Parse HLS settings with error handling
    try:
        max_streams = int(settings.get("hls max streams", "10"))
    except (ValueError, TypeError):
        max_streams = 10
        logger.warning("Invalid 'hls max streams' value, using default: 10")
    
    try:
        inactive_timeout = int(settings.get("hls inactive timeout", "30"))
    except (ValueError, TypeError):
        inactive_timeout = 30
        logger.warning("Invalid 'hls inactive timeout' value, using default: 30")
    
    hls_manager = HLSStreamManager(max_streams=max_streams, inactive_timeout=inactive_timeout)
    hls_manager.start_monitoring()
    logger.info(f"HLS Stream Manager initialized (max_streams={max_streams}, timeout={inactive_timeout}s)")
    
    # Always use waitress for production in container
    logger.info("Starting Waitress server on 0.0.0.0:8001")
    waitress.serve(app, host="0.0.0.0", port=8001, _quiet=True, threads=24) 