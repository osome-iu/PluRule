"""
Shared media extraction and download helpers.

Used by:
  - pipeline/7_collect_media.py  (end-to-end reconstruction, iterates per-subreddit
                                  submission .zst files)
  - hydrate/2_download_media.py  (benchmark hydration, iterates submissions embedded
                                  in hydrated JSON.zst files)

Priority hierarchy for URL extraction (early-stopping):
  1. `media_metadata`  — gallery / inline images (1-N items)
  2. `url`             — direct image posts
  3. `oembed`          — video thumbnails
  4. `preview`         — Reddit-cached preview images

Downloads are validated (Content-Type allowlist) and capped at MAX_FILE_SIZE.
NSFW / crosspost / video submissions are skipped at the top of
`download_submission_media`.
"""

import os
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.files import ensure_directory


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_AGENT = "reddit_research_media_collector/1.0"

# Download limits
DOWNLOAD_TIMEOUT = 15                # seconds per request
MAX_FILE_SIZE = 50 * 1024 * 1024     # 50 MB hard cap
REQUEST_DELAY = 0.1                  # between downloads for one submission

# Domain / extension allowlists
VIDEO_DOMAINS = frozenset([
    'v.redd.it', 'youtube.com', 'youtu.be', 'vimeo.com',
    'streamable.com', 'twitch.tv', 'clips.twitch.tv',
    'tiktok.com', 'instagram.com', 'dailymotion.com',
])

EXTENSIONLESS_MEDIA_HOSTS = frozenset([
    'imgur.com', 'i.imgur.com', 'giphy.com', 'gfycat.com',
])

IMAGE_EXTENSIONS = frozenset(['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'])

VALID_CONTENT_TYPES = frozenset([
    'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp',
])

CONTENT_TYPE_TO_EXT = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/bmp': '.bmp',
}


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def extract_extension_from_url(url: str) -> Optional[str]:
    """Extract image extension from URL path (no dot)."""
    try:
        path = urllib.parse.urlparse(url).path.lower()
        if '.' in path:
            ext = path.split('.')[-1]
            if ext in IMAGE_EXTENSIONS:
                return ext
    except Exception:
        pass
    return None


def is_video_domain(url: str) -> bool:
    try:
        domain = urllib.parse.urlparse(url).netloc.lower()
        return any(vd in domain for vd in VIDEO_DOMAINS)
    except Exception:
        return False


def is_likely_media_url(url: str) -> bool:
    """True if the URL likely points to a downloadable image."""
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()

        if is_video_domain(url) or 'reddit.com/gallery/' in url:
            return False

        if 'reddit.com' in domain and 'i.redd.it' not in domain:
            return False

        if '.' in path and path.split('.')[-1] in IMAGE_EXTENSIONS:
            return True

        return any(host in domain for host in EXTENSIONLESS_MEDIA_HOSTS)
    except Exception:
        return False


def sanitize_media_id(media_id: str, max_length: int = 50) -> str:
    return media_id.replace('|', '_').replace('/', '_').replace('\\', '_')[:max_length]


def categorize_error(error_msg: str) -> str:
    lower = error_msg.lower()
    if '404' in error_msg or 'not found' in lower:
        return '404_not_found'
    if '403' in error_msg or 'forbidden' in lower:
        return '403_forbidden'
    if '429' in error_msg or 'too many' in lower:
        return '429_rate_limited'
    if 'timeout' in lower:
        return 'timeout'
    if 'connection' in lower:
        return 'connection_error'
    if 'ssl' in lower or 'certificate' in lower:
        return 'ssl_error'
    if 'content-type' in lower:
        return 'invalid_content_type'
    return 'other_error'


# ---------------------------------------------------------------------------
# Session / file download
# ---------------------------------------------------------------------------

def create_session() -> requests.Session:
    """HTTP session with bounded retry for transient server errors."""
    session = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({'User-Agent': USER_AGENT})
    return session


def download_file(url: str, output_path: str, session: requests.Session) -> Dict[str, Any]:
    """
    Download `url` to `output_path` with Content-Type validation + size cap.

    Returns: {'success': bool, 'file_size': int, 'extension': str, 'error': str}
    """
    try:
        response = session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').lower().split(';')[0].strip()
        if content_type not in VALID_CONTENT_TYPES:
            return {'success': False, 'error': f'Invalid Content-Type: {content_type}'}

        extension = CONTENT_TYPE_TO_EXT.get(content_type, '.jpg')

        ensure_directory(output_path)
        file_size = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_size += len(chunk)
                    if file_size > MAX_FILE_SIZE:
                        os.remove(output_path)
                        return {'success': False, 'error': 'File too large'}
                    f.write(chunk)

        return {'success': True, 'file_size': file_size, 'extension': extension}

    except requests.exceptions.Timeout:
        if os.path.exists(output_path):
            os.remove(output_path)
        return {'success': False, 'error': 'Timeout'}

    except requests.exceptions.HTTPError as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        return {'success': False, 'error': str(e)}

    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        return {'success': False, 'error': f'Download error: {str(e)[:50]}'}


# ---------------------------------------------------------------------------
# URL extraction (priority hierarchy)
# ---------------------------------------------------------------------------

def is_video_submission(submission: Dict) -> bool:
    if submission.get('is_video'):
        return True
    url = submission.get('url', '')
    if url and is_video_domain(url):
        return True
    media_metadata = submission.get('media_metadata')
    if media_metadata:
        for info in media_metadata.values():
            if info.get('e') in ['Video', 'RedditVideo']:
                return True
    return False


def make_media_item(url: str, media_id: str, source: str, index: Optional[int] = None) -> Dict:
    url = url.replace('&amp;', '&')
    item = {
        'url': url,
        'media_id': media_id,
        'source': source,
        'extension_hint': extract_extension_from_url(url),
    }
    if index is not None:
        item['index'] = index
    return item


def extract_media_metadata_urls(submission: Dict) -> List[Dict]:
    media_metadata = submission.get('media_metadata', {})
    if not media_metadata:
        return []
    urls = []
    for idx, (media_id, info) in enumerate(media_metadata.items()):
        if info.get('e') == 'Image' and 's' in info and 'u' in info['s']:
            urls.append(make_media_item(info['s']['u'], media_id, 'media_metadata', idx))
    return urls


def extract_url_field(submission: Dict) -> List[Dict]:
    url = submission.get('url', '')
    if url and is_likely_media_url(url):
        return [make_media_item(url, 'direct', 'url')]
    return []


def extract_oembed_url(submission: Dict) -> List[Dict]:
    media = submission.get('media') or submission.get('secure_media')
    if media and 'oembed' in media:
        thumb = media['oembed'].get('thumbnail_url')
        if thumb:
            return [make_media_item(thumb, 'oembed', 'oembed')]
    return []


def extract_preview_url(submission: Dict) -> List[Dict]:
    preview = submission.get('preview')
    if not preview or submission.get('is_self'):
        return []
    try:
        url = preview['images'][0]['source'].get('url')
        if url:
            return [make_media_item(url, 'preview', 'preview')]
    except (KeyError, TypeError, IndexError):
        pass
    return []


def extract_download_urls(submission: Dict) -> Tuple[List[Dict], Optional[str]]:
    """Priority hierarchy with early stop."""
    for extractor, source in [
        (extract_media_metadata_urls, 'media_metadata'),
        (extract_url_field, 'url'),
        (extract_oembed_url, 'oembed'),
        (extract_preview_url, 'preview'),
    ]:
        urls = extractor(submission)
        if urls:
            return urls, source
    return [], None


# ---------------------------------------------------------------------------
# Top-level per-submission driver
# ---------------------------------------------------------------------------

def download_submission_media(submission: Dict, media_dir: str,
                              session: requests.Session) -> Dict[str, Any]:
    """
    Download every media URL for one submission.

    Skips NSFW, crosspost, and video submissions at the top.
    Filenames: `{media_dir}/{submission_id}_{media_id}.{ext}` (direct/oembed/preview)
               or `{media_dir}/{submission_id}_{index}_{safe_media_id}.{ext}` (gallery).

    Returns:
        {
          'submission_id':   str,
          'status':          'complete' | 'partial' | 'failed' | 'no_media'
                           | 'skipped_nsfw' | 'skipped_crosspost',
          'files_downloaded': int,
          'file_paths':       List[str],  # absolute paths of successful downloads
          'source':           str | None,
          'is_video':         bool,
          'errors':           List[str],
        }
    """
    submission_id = submission.get('id', 'unknown')

    if submission.get('over_18') or submission.get('over18'):
        return {
            'submission_id': submission_id, 'status': 'skipped_nsfw',
            'files_downloaded': 0, 'file_paths': [], 'errors': [],
        }
    if submission.get('crosspost_parent_list') or submission.get('crosspost_parent'):
        return {
            'submission_id': submission_id, 'status': 'skipped_crosspost',
            'files_downloaded': 0, 'file_paths': [], 'errors': [],
        }

    urls, source = extract_download_urls(submission)
    is_video = is_video_submission(submission)

    if not urls:
        return {
            'submission_id': submission_id, 'status': 'no_media',
            'files_downloaded': 0, 'file_paths': [], 'is_video': is_video,
            'errors': [],
        }

    successful = 0
    file_paths: List[str] = []
    errors: List[str] = []

    for url_info in urls:
        url = url_info['url']
        media_id = url_info['media_id']
        ext_hint = url_info.get('extension_hint')

        if media_id in ('direct', 'oembed', 'preview'):
            filename_base = f"{submission_id}_{media_id}"
        else:
            idx = url_info.get('index', 0)
            safe_id = sanitize_media_id(media_id)
            filename_base = f"{submission_id}_{idx}_{safe_id}"

        if ext_hint:
            cached = os.path.join(media_dir, f"{filename_base}.{ext_hint}")
            if os.path.exists(cached):
                successful += 1
                file_paths.append(cached)
                continue

        temp_path = os.path.join(media_dir, f"{filename_base}.tmp")
        result = download_file(url, temp_path, session)

        if result['success']:
            final_path = os.path.join(media_dir, f"{filename_base}{result['extension']}")
            if os.path.exists(temp_path):
                os.rename(temp_path, final_path)
            successful += 1
            file_paths.append(final_path)
        else:
            errors.append(result['error'])

        time.sleep(REQUEST_DELAY)

    expected = len(urls)
    if successful == expected:
        status = 'complete'
    elif successful > 0:
        status = 'partial'
    else:
        status = 'failed'

    return {
        'submission_id': submission_id,
        'status': status,
        'files_downloaded': successful,
        'file_paths': file_paths,
        'source': source,
        'is_video': is_video,
        'errors': errors,
    }
