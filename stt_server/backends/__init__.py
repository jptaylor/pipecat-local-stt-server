"""Optional backend implementations. Import lazily to avoid hard deps."""

# Whisper repo default for ``--backend mlx``. Lives here (stdlib-only) so the
# CLI can import it without pulling ``mlx_whisper`` (which imports numpy) and
# the backend + prefetch specs share the single definition.
DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
