"""Generate high-resolution (1440x900) UI mock screenshots for Kortex AI documentation."""

import os
from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = "docs/assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# Fonts
FONT_REGULAR = "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

def get_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except:
        return ImageFont.load_default()

f_sans_sm = get_font(FONT_REGULAR, 12)
f_sans_md = get_font(FONT_REGULAR, 14)
f_sans_lg = get_font(FONT_REGULAR, 16)
f_sans_bold_md = get_font(FONT_BOLD, 14)
f_sans_bold_lg = get_font(FONT_BOLD, 18)
f_sans_title = get_font(FONT_BOLD, 42)

f_mono_xs = get_font(FONT_MONO, 11)
f_mono_sm = get_font(FONT_MONO, 12)
f_mono_bold_sm = get_font(FONT_MONO_BOLD, 12)
f_mono_bold_md = get_font(FONT_MONO_BOLD, 14)
f_mono_title = get_font(FONT_MONO_BOLD, 46)

# Swiss Dark Mode Palette
C_CANVAS = "#0C0D10"
C_SURFACE = "#14161E"
C_SURFACE_SUBTLE = "#1A1D28"
C_BORDER = "#2A2F40"
C_BORDER_STRONG = "#3B4257"
C_INK_PRIMARY = "#F4F4F0"
C_INK_SECONDARY = "#9BA0B0"
C_INK_MUTED = "#8E95A5"
C_COBALT = "#3B72FF"
C_COBALT_HOVER = "#2960EF"
C_COBALT_SUBTLE = "#1E294B"
C_EMERALD = "#10B981"
C_EMERALD_SUBTLE = "#0D3320"
C_VERMILION = "#EF4444"
C_AMBER = "#F59E0B"

def draw_header_and_sidebar(draw, active_tab="dossiers"):
    # Sidebar
    draw.rectangle([0, 0, 260, 900], fill=C_SURFACE, outline=C_BORDER)
    # Sidebar Brand
    draw.text((20, 24), "KORTEX AI", fill=C_INK_PRIMARY, font=f_mono_bold_md)
    draw.ellipse([110, 28, 116, 34], fill=C_COBALT)
    draw.text((20, 44), "ENTERPRISE INTELLIGENCE DOSSIER", fill=C_INK_MUTED, font=f_mono_xs)
    draw.line([0, 68, 260, 68], fill=C_BORDER, width=1)

    # Sidebar New Inquiry Button
    draw.rectangle([16, 82, 244, 118], fill=C_INK_PRIMARY)
    draw.text((32, 92), "+ NEW INQUIRY", fill=C_CANVAS, font=f_mono_bold_sm)
    draw.text((210, 92), "⌘N", fill=C_CANVAS, font=f_mono_xs)

    # Sidebar Corpus Button
    draw.rectangle([16, 126, 244, 158], fill=C_SURFACE_SUBTLE, outline=C_BORDER)
    draw.text((32, 136), "02 // KNOWLEDGE CORPUS", fill=C_INK_SECONDARY, font=f_mono_sm)
    draw.text((226, 136), "→", fill=C_INK_MUTED, font=f_mono_sm)
    draw.line([0, 172, 260, 172], fill=C_BORDER, width=1)

    # Inquiries List Header
    draw.text((20, 186), "01 // INQUIRIES DOSSIER", fill=C_INK_MUTED, font=f_mono_xs)
    draw.text((220, 186), "[04]", fill=C_INK_MUTED, font=f_mono_xs)
    draw.line([16, 204, 244, 204], fill=C_BORDER, width=1)

    # Inquiries List
    items = [
        ("01", "Fraud Escalation Protocol", True),
        ("02", "French Glass Breakage Deductibles", False),
        ("03", "Damage Quantification Formula", False),
        ("04", "Tunisian Arabizi Query Screening", False),
    ]
    y = 216
    for num, title, active in items:
        if active:
            draw.rectangle([16, y, 244, y + 36], fill=C_SURFACE_SUBTLE, outline=C_BORDER_STRONG)
            draw.line([16, y, 16, y + 36], fill=C_COBALT, width=3)
            draw.text((26, y + 10), f"{num}  {title[:22]}...", fill=C_INK_PRIMARY, font=f_mono_bold_sm)
        else:
            draw.rectangle([16, y, 244, y + 36], fill=C_SURFACE)
            draw.text((26, y + 10), f"{num}  {title[:22]}...", fill=C_INK_SECONDARY, font=f_mono_sm)
        y += 42

    # Sidebar Telemetry Footer
    draw.line([0, 760, 260, 760], fill=C_BORDER, width=1)
    draw.rectangle([0, 761, 260, 900], fill=C_CANVAS)
    draw.text((20, 776), "03 // SYSTEM ARCHITECTURE", fill=C_INK_MUTED, font=f_mono_xs)
    
    metrics = [
        ("PIPELINE", "HYBRID RAG"),
        ("VDB ENGINE", "CHROMADB (LOCAL)"),
        ("RERANKER", "BGE-RERANKER-V2"),
        ("STATUS", "ONLINE"),
    ]
    my = 798
    for k, v in metrics:
        draw.text((20, my), k, fill=C_INK_MUTED, font=f_mono_xs)
        color = C_EMERALD if k in ("STATUS", "RERANKER") else C_INK_PRIMARY
        draw.text((150, my), v, fill=color, font=f_mono_bold_sm if k == "STATUS" else f_mono_sm)
        my += 22

    # Top Header (Clean minimalist)
    draw.rectangle([260, 0, 1440, 52], fill=C_SURFACE, outline=C_BORDER)
    # Right Actions
    draw.rectangle([1120, 10, 1270, 42], fill=C_SURFACE_SUBTLE, outline=C_BORDER)
    draw.text((1134, 18), "KNOWLEDGE BASE", fill=C_INK_PRIMARY, font=f_mono_bold_sm)

    draw.rectangle([1280, 10, 1354, 42], fill=C_SURFACE, outline=C_BORDER)
    draw.text((1294, 18), "RESET", fill=C_INK_MUTED, font=f_mono_bold_sm)

    draw.rectangle([1364, 10, 1424, 42], fill=C_SURFACE, outline=C_BORDER)
    draw.text((1378, 18), "DARK", fill=C_COBALT, font=f_mono_bold_sm)


# SCREEN 1: KORTEX CHAT WORKSPACE (DARK MODE HOME)
def generate_screen1():
    img = Image.new("RGB", (1440, 900), color=C_CANVAS)
    draw = ImageDraw.Draw(img)
    draw_header_and_sidebar(draw)

    # Main Center Welcome
    cx = 320
    draw.text((cx, 80), "SYSTEM EDITION 02.4 // KORTEX HYBRID RAG", fill=C_INK_MUTED, font=f_mono_sm)
    draw.text((1180, 80), "CHROMA + BM25 + CROSS-ENCODER", fill=C_INK_MUTED, font=f_mono_sm)
    draw.line([cx, 104, 1380, 104], fill=C_BORDER, width=1)

    draw.text((cx, 124), "KORTEX", fill=C_INK_PRIMARY, font=f_mono_title)
    draw.text((cx + 210, 132), "[AI]", fill=C_COBALT, font=f_sans_bold_lg)

    desc = "Autonomous multilingual document intelligence and hybrid retrieval-augmented generation engine.\nFormulate inquiries across policies, contracts, specifications, and verify grounded source citations."
    draw.text((cx, 188), desc, fill=C_INK_SECONDARY, font=f_sans_md, spacing=6)

    # Suggested Inquiries Section
    draw.text((cx, 260), "01 // RECOMMENDED INQUIRIES", fill=C_INK_MUTED, font=f_mono_xs)
    draw.text((1220, 260), "SELECT DOSSIER PROMPT", fill=C_INK_MUTED, font=f_mono_xs)
    draw.line([cx, 280, 1380, 280], fill=C_BORDER, width=1)

    cards = [
        ("[01] FRAUD RISK POLICY", "Fraud Escalation Threshold", "Inspect strict probability boundaries for automated claims suspension.", '"What is the fraud score threshold for human-in-the-loop review?"'),
        ("[02] FRENCH BARÈME", "Glass Breakage Deductibles", "Extract official schedule values and conditional deductible exclusions.", '"Quel est le montant de la franchise bris de glace selon le barème ?"'),
        ("[03] DETERMINISTIC CODE", "Damage Quantification Math", "Verify rule-based algorithms for vehicle component repair estimates.", '"Comment est calculé le chiffrage déterministe des dommages en Python ?"'),
        ("[04] INSTANT BYPASS", "Conversational Screening", "Demonstrate 0ms retrieval avoidance on conversational greetings.", '"Bonjour ! Qui es-tu et quelles sont tes capacités d\'analyse ?"'),
    ]

    for idx, (tag, title, detail, quote) in enumerate(cards):
        col = idx % 2
        row = idx // 2
        x0 = cx + col * 535
        y0 = 300 + row * 160
        x1 = x0 + 515
        y1 = y0 + 144

        draw.rectangle([x0, y0, x1, y1], fill=C_SURFACE, outline=C_BORDER)
        draw.text((x0 + 16, y0 + 14), tag, fill=C_COBALT, font=f_mono_bold_sm)
        draw.text((x1 - 32, y0 + 14), "→", fill=C_INK_MUTED, font=f_mono_bold_sm)
        draw.text((x0 + 16, y0 + 36), title, fill=C_INK_PRIMARY, font=f_mono_bold_md)
        draw.text((x0 + 16, y0 + 60), detail, fill=C_INK_SECONDARY, font=f_sans_sm)
        draw.line([x0 + 16, y1 - 36, x1 - 16, y1 - 36], fill=C_BORDER, width=1)
        draw.text((x0 + 16, y1 - 24), quote, fill=C_INK_MUTED, font=f_mono_xs)

    # Grounded Accuracy Footer
    draw.rectangle([cx, 630, 1380, 680], fill=C_SURFACE_SUBTLE, outline=C_BORDER)
    draw.text((cx + 18, 646), "GROUNDED ACCURACY GUARANTEE:", fill=C_INK_PRIMARY, font=f_mono_bold_sm)
    draw.text((cx + 270, 646), "Zero hallucination via strict context boundary checks.", fill=C_INK_SECONDARY, font=f_sans_sm)
    draw.text((1280, 646), "MANAGE CORPUS →", fill=C_COBALT, font=f_mono_bold_sm)

    # Input Composer Dock
    iy = 720
    draw.text((cx, iy), "PARAMS [▼]", fill=C_INK_SECONDARY, font=f_mono_bold_sm)
    draw.text((cx + 100, iy), "MODE: HYBRID RRF", fill=C_INK_PRIMARY, font=f_mono_sm)
    draw.text((cx + 240, iy), "|", fill=C_BORDER_STRONG, font=f_mono_sm)
    draw.text((cx + 260, iy), "RERANK: CROSS-ENCODER", fill=C_EMERALD, font=f_mono_bold_sm)
    draw.text((1260, iy), "SUBMIT WITH  [↵ ENTER]", fill=C_INK_MUTED, font=f_mono_xs)

    draw.rectangle([cx, iy + 22, 1380, iy + 140], fill=C_SURFACE, outline=C_COBALT, width=2)
    draw.text((cx + 20, iy + 44), "Formulate an inquiry, verify policy clauses, contracts, or technical specifications...", fill=C_INK_MUTED, font=f_sans_md)
    draw.line([cx, iy + 92, 1380, iy + 92], fill=C_BORDER, width=1)
    draw.rectangle([cx, iy + 93, 1380, iy + 140], fill=C_SURFACE_SUBTLE)
    draw.text((cx + 20, iy + 108), "QUERY DOSSIER COMPOSER", fill=C_INK_MUTED, font=f_mono_xs)
    draw.rectangle([1260, iy + 102, 1364, iy + 132], fill=C_COBALT)
    draw.text((1276, iy + 110), "RUN QUERY ↑", fill="#FFFFFF", font=f_mono_bold_sm)

    img.save(os.path.join(ASSETS_DIR, "kortex_chat_workspace_dark.png"), quality=95)
    print("Generated: kortex_chat_workspace_dark.png")


# SCREEN 2: RAG SYNTHESIS & EVIDENCE DOSSIER
def generate_screen2():
    img = Image.new("RGB", (1440, 900), color=C_CANVAS)
    draw = ImageDraw.Draw(img)
    draw_header_and_sidebar(draw)

    cx = 320

    # User Query Bubble
    uy = 75
    draw.rectangle([cx + 280, uy, 1380, uy + 85], fill=C_SURFACE, outline=C_BORDER_STRONG)
    draw.rectangle([cx + 280, uy, 1380, uy + 30], fill=C_SURFACE_SUBTLE)
    draw.text((cx + 296, uy + 8), "INQUIRY // CLIENT • 19:42:01", fill=C_INK_MUTED, font=f_mono_xs)
    draw.text((1330, uy + 8), "[COPY]", fill=C_INK_MUTED, font=f_mono_xs)
    draw.text((cx + 296, uy + 44), "What is the fraud escalation threshold for human-in-the-loop review in automated claims?", fill=C_INK_PRIMARY, font=f_sans_md)

    # Assistant Synthesis Card
    ay = 180
    draw.rectangle([cx, ay, 1380, ay + 520], fill=C_SURFACE, outline=C_BORDER)
    # Header
    draw.rectangle([cx, ay, 1380, ay + 36], fill=C_CANVAS, outline=C_BORDER)
    draw.ellipse([cx + 16, ay + 14, cx + 24, ay + 22], fill=C_COBALT)
    draw.text((cx + 32, ay + 10), "KORTEX AI // SYNTHESIS • 19:42:02", fill=C_INK_PRIMARY, font=f_mono_bold_sm)
    draw.text((1320, ay + 10), "[COPY]", fill=C_INK_MUTED, font=f_mono_sm)

    # Swiss Telemetry Grid
    draw.rectangle([cx, ay + 36, 1380, ay + 74], fill=C_SURFACE_SUBTLE, outline=C_BORDER)
    telemetry = [
        ("INTENT:", "FRAUD_POLICY_SEARCH"),
        ("LANG:", "EN"),
        ("LATENCY:", "412ms"),
        ("TOKENS:", "648"),
        ("EST. COST:", "$0.0013"),
    ]
    tx = cx + 16
    for k, v in telemetry:
        draw.text((tx, ay + 46), k, fill=C_INK_MUTED, font=f_mono_xs)
        tx += len(k) * 8 + 4
        draw.text((tx, ay + 46), v, fill=C_INK_PRIMARY, font=f_mono_bold_sm)
        tx += len(v) * 9 + 24

    # Markdown Prose Body
    py = ay + 96
    draw.text((cx + 24, py), "Fraud Escalation Threshold & Review Protocol", fill=C_INK_PRIMARY, font=f_sans_bold_lg)
    py += 36
    prose_lines = [
        "Based on Section 4.2 of the Underwriting & Fraud Risk Governance Framework:",
        "",
        "• Hard Suspension Boundary (Threshold ≥ 0.70):",
        "  Any claim evaluated with a fraud probability score equal to or exceeding 0.70 is automatically",
        "  halted. Automated disbursement is blocked immediately and quarantined for Tier-3 Special Investigation.",
        "",
        "• Human-in-the-Loop Review Window (0.45 ≤ Score < 0.70):",
        "  Claims in this range require mandatory secondary manual review by an accredited senior claims adjuster",
        "  prior to settlement. A structured audit rationale must be submitted within 24 business hours.",
        "",
        "• Fast-Track Autonomous Processing (Score < 0.45):",
        "  Eligible for zero-touch straight-through payment authorization upon cryptographic validation.",
    ]
    for line in prose_lines:
        color = C_INK_PRIMARY if line.startswith("•") else C_INK_SECONDARY
        font = f_sans_bold_md if line.startswith("•") else f_sans_sm
        draw.text((cx + 24, py), line, fill=color, font=font)
        py += 20

    # Grounded Evidence Dossier
    ey = ay + 370
    draw.rectangle([cx, ey, 1380, ay + 520], fill=C_CANVAS, outline=C_BORDER)
    draw.text((cx + 20, ey + 14), "02 // GROUNDED EVIDENCE DOSSIER (2 PASSAGES)", fill=C_INK_PRIMARY, font=f_mono_bold_sm)
    draw.text((1230, ey + 14), "INSPECT IN DRAWER →", fill=C_COBALT, font=f_mono_bold_sm)

    # 2 Evidence Cards
    cards_ev = [
        ("[REF #01] fraud_escalation_protocol.pdf", "SCORE 0.9412", "Clause 4.2: Automated settlement systems shall enforce an immutable threshold of 0.70. Claims with score >= 0.70 are quarantined..."),
        ("[REF #02] claims_governance_2024.docx", "SCORE 0.8870", "Review Tiers: Claims between 0.45 and 0.69 require human-in-the-loop audit prior to indemnity disbursement. Validation window: 24h."),
    ]
    for idx, (title, score, passage) in enumerate(cards_ev):
        x0 = cx + 20 + idx * 525
        y0 = ey + 40
        x1 = x0 + 510
        y1 = y0 + 90
        draw.rectangle([x0, y0, x1, y1], fill=C_SURFACE, outline=C_BORDER)
        draw.text((x0 + 12, y0 + 10), title, fill=C_COBALT, font=f_mono_bold_sm)
        draw.rectangle([x1 - 100, y0 + 8, x1 - 10, y0 + 26], fill=C_SURFACE_SUBTLE, outline=C_BORDER)
        draw.text((x1 - 94, y0 + 11), score, fill=C_INK_PRIMARY, font=f_mono_xs)
        draw.text((x0 + 12, y0 + 38), passage[:88] + "...", fill=C_INK_SECONDARY, font=f_sans_sm)

    # Dock at bottom
    iy = 720
    draw.rectangle([cx, iy + 22, 1380, iy + 140], fill=C_SURFACE, outline=C_BORDER)
    draw.text((cx + 20, iy + 44), "Ask a follow-up inquiry or test deductible schedule values...", fill=C_INK_MUTED, font=f_sans_md)
    draw.rectangle([cx, iy + 93, 1380, iy + 140], fill=C_SURFACE_SUBTLE)
    draw.text((cx + 20, iy + 108), "QUERY DOSSIER COMPOSER", fill=C_INK_MUTED, font=f_mono_xs)
    draw.rectangle([1260, iy + 102, 1364, iy + 132], fill=C_COBALT)
    draw.text((1276, iy + 110), "RUN QUERY ↑", fill="#FFFFFF", font=f_mono_bold_sm)

    img.save(os.path.join(ASSETS_DIR, "kortex_rag_synthesis_evidence.png"), quality=95)
    print("Generated: kortex_rag_synthesis_evidence.png")


# SCREEN 3: SOURCE VIEWER DRAWER (EXCERPT INSPECTION)
def generate_screen3():
    img = Image.new("RGB", (1440, 900), color=C_CANVAS)
    draw = ImageDraw.Draw(img)
    draw_header_and_sidebar(draw)

    # Dimmed backdrop
    overlay = Image.new("RGBA", (1440, 900), (0, 0, 0, 160))
    img.paste(overlay, (0, 0), overlay)
    draw = ImageDraw.Draw(img)

    # Slide-over Drawer (Right panel 720px wide)
    dx0 = 720
    draw.rectangle([dx0, 0, 1440, 900], fill=C_SURFACE, outline=C_BORDER_STRONG)

    # Drawer Header
    draw.rectangle([dx0, 0, 1440, 70], fill=C_CANVAS, outline=C_BORDER)
    draw.ellipse([dx0 + 24, 26, dx0 + 32, 34], fill=C_COBALT)
    draw.text((dx0 + 40, 22), "EVIDENCE DOSSIER // EXCERPT INSPECTION", fill=C_INK_PRIMARY, font=f_mono_bold_md)
    draw.text((dx0 + 40, 44), "3 PASSAGES RERANKED VIA CROSS-ENCODER", fill=C_INK_MUTED, font=f_mono_xs)
    draw.rectangle([1388, 20, 1420, 50], fill=C_SURFACE, outline=C_BORDER)
    draw.text((1400, 26), "✕", fill=C_INK_MUTED, font=f_mono_bold_sm)

    # Chunks Navigation Tabs
    draw.rectangle([dx0, 70, 1440, 115], fill=C_SURFACE_SUBTLE, outline=C_BORDER)
    # Active tab 1
    draw.rectangle([dx0 + 24, 78, dx0 + 190, 108], fill=C_COBALT)
    draw.text((dx0 + 36, 88), "RANK #1 (0.9412)", fill="#FFFFFF", font=f_mono_bold_sm)
    # Tab 2
    draw.rectangle([dx0 + 200, 78, dx0 + 366, 108], fill=C_SURFACE, outline=C_BORDER)
    draw.text((dx0 + 212, 88), "RANK #2 (0.8870)", fill=C_INK_SECONDARY, font=f_mono_sm)
    # Tab 3
    draw.rectangle([dx0 + 376, 78, dx0 + 542, 108], fill=C_SURFACE, outline=C_BORDER)
    draw.text((dx0 + 388, 88), "RANK #3 (0.8241)", fill=C_INK_SECONDARY, font=f_mono_sm)

    # Selected Chunk Metadata Card
    my = 135
    draw.rectangle([dx0 + 24, my, 1416, my + 175], fill=C_SURFACE_SUBTLE, outline=C_BORDER)
    # Document title
    draw.text((dx0 + 40, my + 16), "📄  fraud_escalation_protocol.pdf", fill=C_INK_PRIMARY, font=f_mono_bold_md)
    draw.rectangle([1310, my + 12, 1400, my + 38], fill=C_SURFACE, outline=C_BORDER)
    draw.text((1324, my + 18), "COPY TEXT", fill=C_INK_SECONDARY, font=f_mono_xs)
    draw.line([dx0 + 24, my + 50, 1416, my + 50], fill=C_BORDER, width=1)

    # Technical Score Breakdown (4 metrics)
    b_y = my + 64
    scores = [
        ("FINAL RELEVANCE", "0.9412", C_COBALT),
        ("RERANK SCORE", "0.9120", C_INK_PRIMARY),
        ("RRF RECIPROCAL", "0.0278", C_INK_PRIMARY),
        ("CHUNK HASH ID", "8f4a1c9d2e...", C_INK_MUTED),
    ]
    bx = dx0 + 40
    for title, val, color in scores:
        draw.text((bx, b_y), title, fill=C_INK_MUTED, font=f_mono_xs)
        draw.text((bx, b_y + 20), val, fill=color, font=f_mono_bold_md)
        bx += 165

    draw.text((dx0 + 40, my + 128), "EMBEDDING MODEL: text-embedding-3-small (1536-dim)", fill=C_INK_SECONDARY, font=f_mono_xs)
    draw.text((dx0 + 40, my + 148), "CROSS-ENCODER: BAAI/bge-reranker-v2-m3", fill=C_EMERALD, font=f_mono_bold_sm)

    # Verbatim Text Section
    vy = 335
    draw.text((dx0 + 24, vy), "VERBATIM PASSAGE TEXT", fill=C_INK_MUTED, font=f_mono_xs)
    draw.text((1320, vy), "492 CHARS", fill=C_INK_MUTED, font=f_mono_xs)

    draw.rectangle([dx0 + 24, vy + 20, 1416, vy + 380], fill=C_CANVAS, outline=C_BORDER)
    raw_text = (
        "SECTION 4.2 - FRAUD PROBABILITY SCORING & HUMAN-IN-THE-LOOP INTERVENTION\n\n"
        "1. AUTOMATED QUARANTINE THRESHOLD:\n"
        "   Any insurance claim where the predictive model produces a composite fraud risk\n"
        "   score P(fraud) >= 0.70 shall be halted immediately. Autonomous disbursement\n"
        "   triggers are deactivated and the dossier is escalated to Tier-3 SIU review.\n\n"
        "2. CONDITIONAL AUDIT WINDOW:\n"
        "   For claims with 0.45 <= P(fraud) < 0.70, straight-through processing is suspended\n"
        "   pending validation by an accredited claims examiner. A formal risk mitigation\n"
        "   rationale must be logged within 24 hours of alert generation."
    )
    draw.text((dx0 + 40, vy + 40), raw_text, fill=C_INK_PRIMARY, font=f_mono_sm, spacing=6)

    # Bottom Actions
    draw.rectangle([dx0, 840, 1440, 900], fill=C_CANVAS, outline=C_BORDER)
    draw.text((dx0 + 24, 862), "KORTEX PROVENANCE ENGINE // AUDIT-READY CITATION", fill=C_INK_MUTED, font=f_mono_xs)
    draw.rectangle([1320, 852, 1416, 888], fill=C_SURFACE, outline=C_BORDER)
    draw.text((1342, 862), "DISMISS", fill=C_INK_PRIMARY, font=f_mono_bold_sm)

    img.save(os.path.join(ASSETS_DIR, "kortex_source_drawer.png"), quality=95)
    print("Generated: kortex_source_drawer.png")


# SCREEN 4: KNOWLEDGE CORPUS MODAL
def generate_screen4():
    img = Image.new("RGB", (1440, 900), color=C_CANVAS)
    draw = ImageDraw.Draw(img)
    draw_header_and_sidebar(draw)

    # Dimmed backdrop
    overlay = Image.new("RGBA", (1440, 900), (0, 0, 0, 180))
    img.paste(overlay, (0, 0), overlay)
    draw = ImageDraw.Draw(img)

    # Modal Box (Centered, 960x740)
    mx0 = 240
    my0 = 80
    mx1 = mx0 + 960
    my1 = my0 + 740

    draw.rectangle([mx0, my0, mx1, my1], fill=C_SURFACE, outline=C_BORDER_STRONG, width=2)

    # Header
    draw.rectangle([mx0, my0, mx1, my0 + 60], fill=C_CANVAS, outline=C_BORDER)
    draw.ellipse([mx0 + 24, my0 + 24, mx0 + 32, my0 + 32], fill=C_COBALT)
    draw.text((mx0 + 42, my0 + 16), "02 // ENTERPRISE KNOWLEDGE CORPUS", fill=C_INK_PRIMARY, font=f_mono_bold_md)
    draw.text((mx0 + 42, my0 + 38), "Corpus document ledger, incremental vector chunking & BM25 inverted index synchronizer.", fill=C_INK_MUTED, font=f_sans_sm)
    draw.text((mx1 - 36, my0 + 20), "✕", fill=C_INK_MUTED, font=f_mono_bold_md)

    # Stats Ribbon (4 blocks)
    sy0 = my0 + 60
    sy1 = sy0 + 68
    draw.rectangle([mx0, sy0, mx1, sy1], fill=C_SURFACE_SUBTLE, outline=C_BORDER)
    stats = [
        ("01 // DOCUMENTS", "12 FILES", C_INK_PRIMARY),
        ("02 // VECTOR CHUNKS", "846 CHUNKS", C_COBALT),
        ("03 // VDB STORAGE", "CHROMADB (LOCAL)", C_INK_PRIMARY),
        ("04 // SYNCHRONIZATION", "ACTIVE ●", C_EMERALD),
    ]
    sx = mx0 + 36
    for k, v, c in stats:
        draw.text((sx, sy0 + 14), k, fill=C_INK_MUTED, font=f_mono_xs)
        draw.text((sx, sy0 + 34), v, fill=c, font=f_mono_bold_md)
        sx += 230

    # Drag & Drop Upload Zone
    uy = sy1 + 20
    draw.text((mx0 + 24, uy), "UPLOAD NEW POLICY OR DOCUMENT", fill=C_INK_MUTED, font=f_mono_xs)
    draw.text((mx1 - 160, uy), "MAX 50MB PER FILE", fill=C_INK_MUTED, font=f_mono_xs)

    draw.rectangle([mx0 + 24, uy + 18, mx1 - 24, uy + 140], fill=C_SURFACE_SUBTLE, outline=C_COBALT, width=1)
    draw.text((mx0 + 430, uy + 36), "☁ UPLOAD", fill=C_COBALT, font=f_mono_bold_md)
    draw.text((mx0 + 340, uy + 64), "Drop files here or click to browse", fill=C_INK_PRIMARY, font=f_sans_bold_md)
    draw.text((mx0 + 320, uy + 88), "Supported formats: PDF, DOCX, TXT, MD, CSV, JSON", fill=C_INK_MUTED, font=f_sans_sm)
    
    # Formats tags
    tags = ["PDF", "DOCX", "TXT", "MD", "CSV", "JSON"]
    tx = mx0 + 330
    for t in tags:
        draw.rectangle([tx, uy + 112, tx + 40, uy + 130], fill=C_SURFACE, outline=C_BORDER)
        draw.text((tx + 6, uy + 115), t, fill=C_INK_MUTED, font=f_mono_xs)
        tx += 48

    # Ingested Corpus Ledger Table
    ly = uy + 160
    draw.text((mx0 + 24, ly), "INGESTED CORPUS LEDGER (4 ENTRIES)", fill=C_INK_PRIMARY, font=f_mono_bold_sm)
    draw.rectangle([mx1 - 180, ly - 6, mx1 - 24, ly + 20], fill=C_SURFACE, outline=C_BORDER)
    draw.text((mx1 - 168, ly - 1), "SYNC ALL ⟳", fill=C_INK_PRIMARY, font=f_mono_bold_sm)

    # Table Header
    ty = ly + 28
    draw.rectangle([mx0 + 24, ty, mx1 - 24, ty + 30], fill=C_SURFACE_SUBTLE, outline=C_BORDER)
    draw.text((mx0 + 36, ty + 8), "DOCUMENT", fill=C_INK_MUTED, font=f_mono_xs)
    draw.text((mx0 + 340, ty + 8), "FORMAT", fill=C_INK_MUTED, font=f_mono_xs)
    draw.text((mx0 + 440, ty + 8), "SIZE", fill=C_INK_MUTED, font=f_mono_xs)
    draw.text((mx0 + 540, ty + 8), "CHUNKS", fill=C_INK_MUTED, font=f_mono_xs)
    draw.text((mx0 + 670, ty + 8), "LAST SYNCED", fill=C_INK_MUTED, font=f_mono_xs)
    draw.text((mx1 - 80, ty + 8), "ACTION", fill=C_INK_MUTED, font=f_mono_xs)

    # Table Rows
    rows = [
        ("fraud_escalation_protocol.pdf", "PDF", "1.4 MB", "48", "Nov 1, 18:30"),
        ("claims_underwriting_guide_v4.docx", "DOCX", "820 KB", "64", "Nov 1, 18:31"),
        ("french_bareme_deductibles.csv", "CSV", "145 KB", "18", "Nov 1, 18:32"),
        ("damage_quantification_rules.py", "TXT", "42 KB", "12", "Nov 1, 18:33"),
    ]
    ry = ty + 30
    for doc, fmt, size, ch, synced in rows:
        draw.rectangle([mx0 + 24, ry, mx1 - 24, ry + 40], fill=C_SURFACE, outline=C_BORDER)
        draw.text((mx0 + 36, ry + 12), f"📄 {doc}", fill=C_INK_PRIMARY, font=f_mono_bold_sm)
        draw.text((mx0 + 340, ry + 12), fmt, fill=C_INK_MUTED, font=f_mono_xs)
        draw.text((mx0 + 440, ry + 12), size, fill=C_INK_SECONDARY, font=f_sans_sm)
        draw.text((mx0 + 540, ry + 12), ch, fill=C_COBALT, font=f_mono_bold_sm)
        draw.text((mx0 + 670, ry + 12), synced, fill=C_INK_MUTED, font=f_mono_xs)
        draw.text((mx1 - 70, ry + 12), "🗑", fill=C_VERMILION, font=f_mono_sm)
        ry += 40

    # Modal Footer
    draw.rectangle([mx0, my1 - 50, mx1, my1], fill=C_CANVAS, outline=C_BORDER)
    draw.text((mx0 + 24, my1 - 32), "KORTEX VDB LEDGER ENGINE // LOCAL PERSISTENCE", fill=C_INK_MUTED, font=f_mono_xs)
    draw.rectangle([mx1 - 120, my1 - 42, mx1 - 24, my1 - 12], fill=C_SURFACE, outline=C_BORDER)
    draw.text((mx1 - 96, my1 - 32), "DISMISS", fill=C_INK_PRIMARY, font=f_mono_bold_sm)

    img.save(os.path.join(ASSETS_DIR, "kortex_knowledge_corpus_modal.png"), quality=95)
    print("Generated: kortex_knowledge_corpus_modal.png")

if __name__ == "__main__":
    generate_screen1()
    generate_screen2()
    generate_screen3()
    generate_screen4()
