"""
modules/fine_report.py  –  Member 4
======================================
Generates professional PDF incident/fine reports using ReportLab with visual evidence frames.
Falls back to plain-text reports if ReportLab is not installed.

Public API
----------
  IncidentReportGenerator        – main class
  build_sample_incident(demo)    – returns a demo incident dict
"""

from __future__ import annotations
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    INSTITUTION,
    DISCLAIMER,
    FINE_AMOUNTS,
    TRASHNET_CLASSES,
    LITTERING_LABELS,
    REPORTS_DIR,
)

# ── Optional: ReportLab ───────────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether, Image as RLImage,
    )
    _RL_AVAILABLE = True
except ImportError:
    _RL_AVAILABLE = False


# ── Sample incident builder ───────────────────────────────────────────────────

def build_sample_incident(demo: bool = True) -> dict:
    import random
    random.seed(99)

    # Use sample crops if available, else blank
    sample_waste_img = ROOT / "sample_data" / "synthetic_waste_images" / "plastic_sample.png"
    sample_person_img = ROOT / "sample_data" / "student_db" / "MITS_STU_2026_01_Anoop.jpg"

    return {
        "littering_event": {
            "summary_event": {
                "event_type":  "potential_littering",
                "confidence":  round(random.uniform(0.72, 0.95), 3),
                "timestamp":   "00:00:14",
                "frame_index": 14,
                "frame_path":  str(sample_waste_img) if sample_waste_img.exists() else "",
            },
            "video_path":    "sample_data/synthetic_video.avi",
            "model_used":    "demo_simulation" if demo else "motion_heuristic",
            "demo_mode":     demo,
            "evidence_path": str(sample_waste_img) if sample_waste_img.exists() else "",
        },
        "waste_result": {
            "predicted_class": "plastic",
            "waste_category":  "non_biodegradable",
            "confidence":      round(random.uniform(0.78, 0.95), 3),
            "demo_mode":       demo,
        },
        "track_info": {
            "track_id":        3,
            "identity_status": "verified" if demo else "unverified",
            "student_name":    "Anoop (MITS-STU-2026-01)" if demo else "Unverified Person",
            "crop_path":       str(sample_person_img) if sample_person_img.exists() else "",
            "first_seen":      8.2,
            "last_seen":       22.7,
            "frame_count":     8,
        },
        "location":            "Main Gate – Camera 02 (Demo)",
        "notes":               "Automated draft report with dual visual evidence captures.",
        "demo_mode":           demo,
    }


# ── Report generator ──────────────────────────────────────────────────────────

class IncidentReportGenerator:

    def __init__(self):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_report_id() -> str:
        ts   = time.strftime("%Y%m%d-%H%M%S")
        uid  = uuid.uuid4().hex[:6].upper()
        return f"GAVA-{ts}-{uid}"

    def generate(self, incident: dict, output_path: Optional[str] = None) -> str:
        report_id = self._make_report_id()

        if output_path is None:
            suffix = ".pdf" if _RL_AVAILABLE else ".txt"
            output_path = str(REPORTS_DIR / f"{report_id}{suffix}")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if _RL_AVAILABLE:
            try:
                return self._generate_pdf(incident, report_id, out)
            except Exception as exc:
                txt_path = out.with_suffix(".txt")
                return self._fallback_text_report(incident, txt_path, report_id, str(exc))
        else:
            return self._fallback_text_report(incident, out, report_id)

    def _generate_pdf(self, incident: dict, report_id: str, out: Path) -> str:
        doc = SimpleDocTemplate(
            str(out),
            pagesize=A4,
            leftMargin=1.5*cm,
            rightMargin=1.5*cm,
            topMargin=1.5*cm,
            bottomMargin=1.5*cm
        )
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=15, spaceAfter=2)
        head_style  = ParagraphStyle("Head", parent=styles["Heading2"], fontSize=10, textColor=colors.HexColor("#2c3e50"))
        body_style  = styles["BodyText"]
        small_style = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=7.5, textColor=colors.grey)
        warn_style  = ParagraphStyle("Warn", parent=styles["BodyText"], fontSize=8.5, textColor=colors.red, spaceBefore=2)

        story = []

        # 1. Institutional Header
        story.append(Paragraph(INSTITUTION["name"], title_style))
        story.append(Paragraph(INSTITUTION["project"], head_style))
        story.append(Paragraph(INSTITUTION["location"], small_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2c3e50")))
        story.append(Spacer(1, 0.2*cm))

        # 2. Metadata Table
        is_demo = incident.get("demo_mode", False)
        stamp   = time.strftime("%Y-%m-%d %H:%M:%S")
        meta_data = [
            ["Report ID", report_id, "Generated", stamp],
            ["Location",  incident.get("location", "—"), "Status", "⚠ DRAFT / NOT VERIFIED" if is_demo else "PENDING REVIEW"],
        ]
        meta_tbl = Table(meta_data, colWidths=[3*cm, 6*cm, 2.5*cm, 6.5*cm])
        meta_tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("PADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(meta_tbl)
        story.append(Spacer(1, 0.25*cm))

        if is_demo:
            story.append(Paragraph(
                "⚠ DEMO / SIMULATION – Generated for testing purposes. Real enforcement requires human review.",
                warn_style
            ))
            story.append(Spacer(1, 0.2*cm))

        # 3. Analytics Results Summary Table
        lit = incident.get("littering_event", {})
        summ = lit.get("summary_event", {})
        wr = incident.get("waste_result", {})
        trk = incident.get("track_info", {})
        cat = wr.get("waste_category", "unknown")
        fine = FINE_AMOUNTS.get(cat, FINE_AMOUNTS["unknown"])

        analytics_data = [
            ["Littering Event", summ.get("event_type", "—"), "Confidence", f"{summ.get('confidence', 0)*100:.1f}%"],
            ["Timestamp", summ.get("timestamp", "—"), "Model", lit.get("model_used", "motion_heuristic")],
            ["Waste Class", wr.get("predicted_class", "—"), "Category", cat],
            ["Waste Confidence", f"{wr.get('confidence', 0)*100:.1f}%", "Proposed Fine", f"INR {fine:,}"],
            ["Track ID", str(trk.get("track_id", "—")), "Identity Status", trk.get("identity_status", "unverified").upper()],
            ["Identified Subject", trk.get("student_name", "Unverified Person"), "Review Status", incident.get("review_status", "Pending Verification")],
        ]
        analytics_tbl = Table(analytics_data, colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 5.5*cm])
        analytics_tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("PADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(analytics_tbl)
        story.append(Spacer(1, 0.3*cm))

        # 4. Visual Evidence Dossier Section
        story.append(Paragraph("<b>4. Visual Evidence Dossier</b>", head_style))
        story.append(Spacer(1, 0.15*cm))

        # Resolve image file paths
        litter_path = summ.get("frame_path") or lit.get("evidence_path") or incident.get("evidence_path")
        person_path = trk.get("crop_path") or incident.get("person_crop_path")

        def _resolve_image_flowable(p_str: Optional[str]):
            if p_str and Path(p_str).exists():
                try:
                    return RLImage(str(p_str), width=8.5*cm, height=6.0*cm)
                except Exception:
                    pass
            return Paragraph("<i>[Visual frame capture unavailable]</i>", small_style)

        ev_img_litter = _resolve_image_flowable(litter_path)
        ev_img_person = _resolve_image_flowable(person_path)

        evidence_matrix = [
            [
                Paragraph("<b>Figure A: Littering / Disposal Event</b>", small_style),
                Paragraph("<b>Figure B: Detected Subject / Identity Crop</b>", small_style)
            ],
            [ev_img_litter, ev_img_person]
        ]

        ev_table = Table(evidence_matrix, colWidths=[9.0*cm, 9.0*cm])
        ev_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(ev_table)
        story.append(Spacer(1, 0.3*cm))

        # 5. Notes & Legal Disclaimer
        notes = incident.get("notes", "")
        if notes:
            story.append(Paragraph(f"<b>Notes:</b> {notes}", small_style))
            story.append(Spacer(1, 0.2*cm))

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Paragraph(DISCLAIMER, small_style))
        story.append(Paragraph(f"Generated by GAVA AI  |  {INSTITUTION['contact']}", small_style))

        doc.build(story)
        return str(out)

    def _fallback_text_report(self, incident: dict, out: Path,
                               report_id: str = "", error: str = "") -> str:
        if not report_id:
            report_id = self._make_report_id()
        lines = [
            "=" * 70,
            f"  {INSTITUTION['name']}",
            f"  {INSTITUTION['project']}",
            "=" * 70,
            f"  Report ID  : {report_id}",
            f"  Generated  : {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Location   : {incident.get('location', '—')}",
            "",
        ]
        if incident.get("demo_mode"):
            lines += ["  *** DEMO / SIMULATION – NOT A REAL INCIDENT ***", ""]

        lit   = incident.get("littering_event", {})
        summ  = lit.get("summary_event", {})
        wr    = incident.get("waste_result", {})
        trk   = incident.get("track_info",   {})
        cat   = wr.get("waste_category", "unknown")
        fine  = FINE_AMOUNTS.get(cat, FINE_AMOUNTS["unknown"])

        lines += [
            "LITTERING DETECTION",
            f"  Event Type : {summ.get('event_type', '—')}",
            f"  Confidence : {summ.get('confidence', 0)*100:.1f}%",
            f"  Timestamp  : {summ.get('timestamp', '—')}",
            "",
            "WASTE CLASSIFICATION",
            f"  Class      : {wr.get('predicted_class', '—')}",
            f"  Category   : {cat}",
            f"  Confidence : {wr.get('confidence', 0)*100:.1f}%",
            f"  Fine       : INR {fine:,}",
            "",
            "PERSON TRACKING & IDENTITY",
            f"  Track ID   : {trk.get('track_id', '—')}",
            f"  Identity   : {trk.get('identity_status', 'unverified').upper()}",
            f"  Subject    : {trk.get('student_name', 'Unverified Person')}",
            "",
            "EVIDENCE CAPTURES",
            f"  Litter Frame : {summ.get('frame_path', 'N/A')}",
            f"  Person Crop  : {trk.get('crop_path', 'N/A')}",
            "",
            "DISCLAIMER",
            DISCLAIMER,
            "",
            "=" * 70,
        ]
        if error:
            lines += [f"[PDF ERROR]: {error}", ""]

        out.parent.mkdir(parents=True, exist_ok=True)
        txt_out = out if out.suffix == ".txt" else out.with_suffix(".txt")
        txt_out.write_text("\n".join(lines), encoding="utf-8")
        return str(txt_out)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GAVA AI – Fine Report Generation (Member 4)")
    parser.add_argument("--incident_json", default="", help="Path to incident JSON file")
    parser.add_argument("--demo", action="store_true", help="Generate a demo report")
    args = parser.parse_args()

    gen = IncidentReportGenerator()

    if args.incident_json and Path(args.incident_json).exists():
        with open(args.incident_json, encoding="utf-8") as f:
            incident = json.load(f)
    else:
        print("[DEMO MODE] Using sample incident …")
        incident = build_sample_incident(demo=True)

    path = gen.generate(incident)
    print(f"Report saved: {path}")
    print(f"ReportLab available: {_RL_AVAILABLE}")