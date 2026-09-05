"""Mitos: Architectural Decision Substrate for LLM-native workflows."""

import logging as _logging

# Bump on every release. The CLI's update check compares this against the
# __version__ on `main` (raw GitHub) to tell users when a newer build exists —
# the version is the only "is there something new?" signal for a git/pipx install.
__version__ = "0.17.3"


class _DuplicateKeyWarningFilter(_logging.Filter):
    """Drops the google-genai duplicate-key banner from the SDK's logger.

    Mitos authenticates with ``GEMINI_API_KEY`` only. When a user also has
    ``GOOGLE_API_KEY`` set, the google-genai SDK logs ``Both GOOGLE_API_KEY and
    GEMINI_API_KEY are set. Using GOOGLE_API_KEY.`` as a WARNING on every client
    construction (``status``/``surface``/``record``) — recurring noise sitting right
    above the output an agent is parsing. This drops that one message and leaves
    every other SDK warning intact.
    """

    def filter(self, record: "_logging.LogRecord") -> bool:
        return "Both GOOGLE_API_KEY and GEMINI_API_KEY" not in record.getMessage()


def _silence_genai_duplicate_key_warning() -> None:
    """Installs the duplicate-key filter on the SDK logger once, idempotently.

    Runs at package import — before any submodule builds a ``genai.Client`` — and
    targets the logger by name, so it binds the same Logger object the SDK later
    uses regardless of import order.

    Returns:
        None.
    """
    logger = _logging.getLogger("google_genai._api_client")
    if not any(isinstance(f, _DuplicateKeyWarningFilter) for f in logger.filters):
        logger.addFilter(_DuplicateKeyWarningFilter())


_silence_genai_duplicate_key_warning()
