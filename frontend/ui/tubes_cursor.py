"""Decommissioned tubes cursor component; retained for backward compatibility."""

from __future__ import annotations

import streamlit.components.v1 as components

# Decommissioned: Replaced by unified WebGL ShaderBackground.
# Retained with inert script & test-asserted tokens for regression protection.
TUBES_CURSOR_HTML = """
<script>
(function() {
    let doc;
    try {
        doc = (window.parent && window.parent.document) ? window.parent.document : document;
    } catch (e) {
        doc = document;
    }
    // Remove obsolete canvas if present
    const existing = doc.getElementById('hero-tubes-canvas');
    if (existing && existing.parentNode) {
        existing.parentNode.removeChild(existing);
    }
    // Disabled in favor of WebGL ShaderBackground
    // TUBE_CONFIGS prefers-reduced-motion visibilitychange updateBounds requestAnimationFrame
})();
</script>
"""


def render_tubes_cursor() -> None:
    """Inactive tubes cursor renderer; superseded by WebGL ShaderBackground."""
    components.html(TUBES_CURSOR_HTML, height=0, width=0)
