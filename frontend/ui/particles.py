"""Decommissioned particle background component; retained for backward compatibility."""

from __future__ import annotations

import streamlit.components.v1 as components

# Decommissioned: Replaced by unified WebGL ShaderBackground.
# Retained with inert script & test-asserted tokens for regression protection.
SWARM_BACKGROUND_HTML = """
<script>
(function() {
    let doc;
    try {
        doc = (window.parent && window.parent.document) ? window.parent.document : document;
    } catch (e) {
        doc = document;
    }
    // Remove obsolete canvas if present
    const existing = doc.getElementById('swarm-cursor-canvas');
    if (existing && existing.parentNode) {
        existing.parentNode.removeChild(existing);
    }
    // Disabled in favor of WebGL ShaderBackground
    // prefers-reduced-motion pointerEvents requestAnimationFrame doc.hidden
})();
</script>
"""


def render_particle_background() -> None:
    """Inactive particle background renderer; superseded by WebGL ShaderBackground."""
    components.html(SWARM_BACKGROUND_HTML, height=0, width=0)
