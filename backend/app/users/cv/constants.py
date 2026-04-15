# 3 MB limit in bytes
MAX_CV_SIZE_BYTES = 3 * 1024 * 1024

# Allow ~1MB multipart overhead to avoid false positives on header-based checks (legacy/multipart context)
MAX_MULTIPART_OVERHEAD_BYTES = 1 * 1024 * 1024

# Allowed content types and extensions
ALLOWED_MIME_TYPES = {
    "text/plain",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}

# Markdown conversion character limit (business rule; enforced in service)
MAX_MARKDOWN_CHARS = 50000

# Markdown conversion timeout in seconds
MARKDOWN_CONVERSION_TIMEOUT_SECONDS = 60

# Defaults for limits when envs are not provided
DEFAULT_MAX_UPLOADS_PER_USER = 3
DEFAULT_RATE_LIMIT_PER_MINUTE = 2
RATE_LIMIT_WINDOW_MINUTES = 1

# Maximum number of completed CV uploads returned in a single listing query.
# Keeps memory usage predictable; users are unlikely to need more than this.
MAX_USER_UPLOADS_RETURNED = 50
