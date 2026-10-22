"""Subtle, responsive, performant WebGL ShaderBackground animated component for Light Editorial workspace."""

from __future__ import annotations

import streamlit.components.v1 as components

HERO_SHADER_HTML = """
<script>
(function() {
    let doc;
    let win;
    try {
        if (window.parent && window.parent.document) {
            doc = window.parent.document;
            win = window.parent;
        } else {
            doc = document;
            win = window;
        }
    } catch (e) {
        doc = document;
        win = window;
    }

    // Clean up any previous background animation loop or listeners before initializing
    if (win.__ai_paprion_shader_cleanup) {
        try {
            win.__ai_paprion_shader_cleanup();
        } catch (e) {}
        win.__ai_paprion_shader_cleanup = null;
    }

    // Completely remove old particle canvas if present in DOM to prevent duplicate effects
    try {
        const oldParticle = doc.getElementById('swarm-cursor-canvas');
        if (oldParticle && oldParticle.parentNode) {
            oldParticle.parentNode.removeChild(oldParticle);
        }
    } catch (e) {}

    let animId = null;
    let gl = null;
    let program = null;
    let posBuffer = null;
    let vs = null;
    let fs = null;
    let uTimeLoc = null;
    let uResLoc = null;
    let startTime = null;
    let lastFrameTime = 0;
    let isReducedMotion = false;
    let isMobile = false;
    let retries = 0;

    function checkReducedMotion() {
        try {
            const mq = win.matchMedia && win.matchMedia('(prefers-reduced-motion: reduce)');
            return mq ? mq.matches : false;
        } catch (e) {
            return false;
        }
    }

    function checkMobile() {
        try {
            return (win.innerWidth <= 768) || ('ontouchstart' in win);
        } catch (e) {
            return false;
        }
    }

    function updateBounds() {
        const mount = doc.getElementById('hero-shader-mount');
        if (!mount) return;
        const searchSec = doc.getElementById('search-section');
        if (searchSec) {
            const searchRow = searchSec.nextElementSibling;
            if (searchRow) {
                const mountRect = mount.getBoundingClientRect();
                const rowRect = searchRow.getBoundingClientRect();
                const desiredHeight = Math.max(180, Math.round(rowRect.bottom - mountRect.top + 16));
                mount.style.height = desiredHeight + 'px';
            }
        }
    }

    function cleanup() {
        if (animId) {
            win.cancelAnimationFrame(animId);
            animId = null;
        }
        win.removeEventListener('resize', onResize);
        win.removeEventListener('orientationchange', onResize);
        doc.removeEventListener('visibilitychange', onVisibilityChange);

        if (gl && program) {
            try {
                if (posBuffer) gl.deleteBuffer(posBuffer);
                if (vs) gl.deleteShader(vs);
                if (fs) gl.deleteShader(fs);
                gl.deleteProgram(program);
            } catch (e) {}
        }
    }

    win.__ai_paprion_shader_cleanup = cleanup;

    function onVisibilityChange() {
        if (doc.hidden) {
            if (animId) {
                win.cancelAnimationFrame(animId);
                animId = null;
            }
        } else if (!isReducedMotion) {
            lastFrameTime = performance.now();
            animId = win.requestAnimationFrame(renderFrame);
        }
    }

    function onResize() {
        isMobile = checkMobile();
        updateBounds();
        const canvas = doc.getElementById('hero-shader-canvas');
        if (!canvas || !gl) return;
        const rect = canvas.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return;
        const maxDpr = isMobile ? 1.0 : 1.5;
        const dpr = Math.min(win.devicePixelRatio || 1, maxDpr);
        const w = Math.max(10, Math.round(rect.width * dpr));
        const h = Math.max(10, Math.round(rect.height * dpr));
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
            gl.viewport(0, 0, w, h);
        }
    }

    function initShader() {
        let mount = doc.getElementById('hero-shader-mount');
        const block = doc.querySelector('[data-testid="stAppViewContainer"] .block-container') || doc.querySelector('.block-container');

        if (!mount) {
            if (retries++ < 60) {
                setTimeout(initShader, 50);
            }
            return;
        }

        if (block && mount.parentElement !== block) {
            block.insertBefore(mount, block.firstChild);
        }

        const canvas = doc.getElementById('hero-shader-canvas');
        if (!canvas) {
            if (retries++ < 60) {
                setTimeout(initShader, 50);
            }
            return;
        }

        if (canvas.__shader_running) {
            updateBounds();
            return;
        }
        canvas.__shader_running = true;

        isReducedMotion = checkReducedMotion();
        isMobile = checkMobile();
        updateBounds();

        try {
            gl = canvas.getContext('webgl', { alpha: false, depth: false, antialias: false, powerPreference: 'low-power' }) ||
                 canvas.getContext('experimental-webgl', { alpha: false, depth: false, antialias: false });
        } catch (e) {
            gl = null;
        }

        if (!gl) {
            init2DFallback(canvas);
            return;
        }

        const vsSource = `
            attribute vec2 a_position;
            varying vec2 v_uv;
            void main() {
                v_uv = (a_position + 1.0) * 0.5;
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;

        const fsSource = `
            precision mediump float;
            uniform vec2 u_resolution;
            uniform float u_time;
            varying vec2 v_uv;

            vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
            vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
            vec3 permute(vec3 x) { return mod289(((x * 34.0) + 1.0) * x); }

            float snoise(vec2 v) {
                const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
                vec2 i  = floor(v + dot(v, C.yy));
                vec2 x0 = v - i + dot(i, C.xx);
                vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
                vec4 x12 = x0.xyxy + C.xxzz;
                x12.xy -= i1;
                i = mod289(i);
                vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
                vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
                m = m * m;
                m = m * m;
                vec3 x = 2.0 * fract(p * C.www) - 1.0;
                vec3 h = abs(x) - 0.5;
                vec3 ox = floor(x + 0.5);
                vec3 a0 = x - ox;
                m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
                vec3 g;
                g.x  = a0.x  * x0.x  + h.x  * x0.y;
                g.yz = a0.yz * x12.xz + h.yz * x12.yw;
                return 130.0 * dot(m, g);
            }

            void main() {
                vec2 st = gl_FragCoord.xy / u_resolution.xy;
                float aspect = u_resolution.x / max(u_resolution.y, 1.0);
                vec2 p = vec2(st.x * aspect, st.y);

                // Very gentle, calm drift speed
                float t = u_time * 0.028;

                // Multi-scale organic noise field without harsh wave contours
                vec2 q = vec2(
                    snoise(p * 0.60 + vec2(t * 0.08, t * 0.04)),
                    snoise(p * 0.60 + vec2(t * 0.05, -t * 0.06))
                );

                vec2 r = vec2(
                    snoise(p * 0.80 + q * 0.75 + vec2(2.1, 5.7) + 0.04 * t),
                    snoise(p * 0.80 + q * 0.75 + vec2(6.4, 1.9) - 0.03 * t)
                );

                float f = snoise(p * 0.70 + r * 0.85 + 0.02 * t);
                f = clamp(f * 0.5 + 0.5, 0.0, 1.0);
                float f2 = clamp(q.x * 0.5 + 0.5, 0.0, 1.0);
                float f3 = clamp(r.y * 0.5 + 0.5, 0.0, 1.0);

                // Calibrated Aurora Glass palette seamlessly blending into --bg-workspace (#F5F4F9)
                vec3 cWorkspace = vec3(0.961, 0.957, 0.976); // #F5F4F9
                vec3 cAmbient   = vec3(0.978, 0.975, 0.988); // #F9F8FD
                vec3 cAuroraLav = vec3(0.925, 0.905, 0.975); // Soft atmospheric lavender
                vec3 cAuroraVio = vec3(0.660, 0.580, 0.890); // Muted atmospheric violet
                vec3 cAuroraBlu = vec3(0.550, 0.680, 0.920); // Restrained atmospheric blue

                // Smooth organic luminous blending without stripes or sharp edges
                vec3 col = mix(cWorkspace, cAmbient, smoothstep(0.12, 0.68, f));
                col = mix(col, cAuroraLav, smoothstep(0.30, 0.82, f2) * 0.28);
                col = mix(col, cAuroraVio, smoothstep(0.48, 0.94, f3) * 0.12);
                col = mix(col, cAuroraBlu, smoothstep(0.55, 0.98, f * f2) * 0.08);

                // Soft vignette and bottom fade for seamless transition into page content
                float vig = 1.0 - smoothstep(0.65, 1.45, length(st - vec2(0.5, 0.4)));
                col = mix(cWorkspace, col, clamp(vig, 0.65, 1.0));

                gl_FragColor = vec4(col, 1.0);
            }
        `;

        function compileShader(type, source) {
            const s = gl.createShader(type);
            gl.shaderSource(s, source);
            gl.compileShader(s);
            if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
                gl.deleteShader(s);
                return null;
            }
            return s;
        }

        vs = compileShader(gl.VERTEX_SHADER, vsSource);
        fs = compileShader(gl.FRAGMENT_SHADER, fsSource);
        if (!vs || !fs) {
            init2DFallback(canvas);
            return;
        }

        program = gl.createProgram();
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            init2DFallback(canvas);
            return;
        }

        gl.useProgram(program);

        posBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
        gl.bufferData(
            gl.ARRAY_BUFFER,
            new Float32Array([
                -1, -1,
                 1, -1,
                -1,  1,
                -1,  1,
                 1, -1,
                 1,  1
            ]),
            gl.STATIC_DRAW
        );

        const posLoc = gl.getAttribLocation(program, 'a_position');
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

        uTimeLoc = gl.getUniformLocation(program, 'u_time');
        uResLoc = gl.getUniformLocation(program, 'u_resolution');

        win.addEventListener('resize', onResize, { passive: true });
        win.addEventListener('orientationchange', onResize, { passive: true });
        doc.addEventListener('visibilitychange', onVisibilityChange, { passive: true });
        onResize();

        startTime = performance.now();
        lastFrameTime = startTime;

        function renderFrame(now) {
            if (!gl || !program) return;

            // Frame throttling: 30fps cap on mobile to conserve GPU and battery
            if (isMobile) {
                const delta = now - lastFrameTime;
                if (delta < 32.0) {
                    animId = win.requestAnimationFrame(renderFrame);
                    return;
                }
                lastFrameTime = now;
            }

            const elapsed = isReducedMotion ? 1.5 : (now - startTime) * 0.001;

            gl.uniform1f(uTimeLoc, elapsed);
            gl.uniform2f(uResLoc, canvas.width, canvas.height);
            gl.drawArrays(gl.TRIANGLES, 0, 6);

            if (!isReducedMotion && !doc.hidden) {
                animId = win.requestAnimationFrame(renderFrame);
            }
        }

        if (isReducedMotion) {
            renderFrame(startTime + 1500);
        } else {
            animId = win.requestAnimationFrame(renderFrame);
        }
    }

    function init2DFallback(canvas) {
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        function drawFallback() {
            updateBounds();
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
            const grad = ctx.createLinearGradient(0, 0, rect.width, rect.height);
            grad.addColorStop(0, '#F5F4F9');
            grad.addColorStop(0.35, '#EDE8FD');
            grad.addColorStop(0.70, '#E0EAFF');
            grad.addColorStop(1, '#F5F4F9');
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, rect.width, rect.height);
        }
        drawFallback();
        win.addEventListener('resize', drawFallback, { passive: true });
        win.addEventListener('orientationchange', drawFallback, { passive: true });
    }

    if (doc.readyState === 'loading') {
        doc.addEventListener('DOMContentLoaded', initShader);
    } else {
        initShader();
    }
})();
</script>
"""


def render_hero_shader() -> None:
    """Render WebGL animated ShaderBackground component for the research hero."""
    components.html(HERO_SHADER_HTML, height=0, width=0)
