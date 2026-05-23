from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "evals" / "results"
REPORTS_DIR = ROOT / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"


EVAL_METRICS = [
    ("overall_pass_rate", "Overall Pass", True),
    ("hallucination_rate", "Hallucination", False),
    ("bias_harmful_output_rate", "Bias/Harm", False),
    ("jailbreak_success_rate", "Jailbreak Success", False),
    ("safety_compliance_failure_rate", "Safety Failure", False),
    ("operational_failure_rate", "Ops Failure", False),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def pct(value: float) -> str:
    return f"{value:.0%}"


def ms(value: float) -> str:
    return f"{value / 1000:.2f}s"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_grouped_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    series: dict[str, list[float]],
    y_max: float = 1.0,
    y_label: str = "Rate",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 760
    margin_left, margin_right, margin_top, margin_bottom = 120, 60, 145, 130
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(34, bold=True)
    label_font = font(18)
    small_font = font(15)

    palette = {
        "Raw OSS": "#64748b",
        "OSS + Guardrails": "#2563eb",
        "Frontier": "#16a34a",
    }
    draw.text((margin_left, 28), title, fill="#111827", font=title_font)
    draw.text((margin_left, 68), y_label, fill="#4b5563", font=label_font)

    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    for i in range(6):
        y_value = y_max * i / 5
        y = plot_bottom - (y_value / y_max) * plot_height
        draw.line((plot_left, y, plot_right, y), fill="#e5e7eb", width=1)
        draw.text((34, y - 10), pct(y_value), fill="#6b7280", font=small_font)

    group_width = plot_width / len(labels)
    names = list(series)
    bar_width = min(46, group_width / (len(names) + 1.5))
    for label_index, label in enumerate(labels):
        group_center = plot_left + group_width * label_index + group_width / 2
        start_x = group_center - (len(names) * bar_width + (len(names) - 1) * 12) / 2
        for series_index, name in enumerate(names):
            value = series[name][label_index]
            x0 = start_x + series_index * (bar_width + 12)
            x1 = x0 + bar_width
            y0 = plot_bottom - (value / y_max) * plot_height
            draw.rounded_rectangle((x0, y0, x1, plot_bottom), radius=5, fill=palette.get(name, "#64748b"))
            draw.text((x0 - 4, y0 - 23), pct(value), fill="#111827", font=small_font)
        text_x = group_center - min(80, len(label) * 5)
        draw.text((text_x, plot_bottom + 18), label, fill="#374151", font=small_font)

    legend_x = plot_right - 235
    for index, name in enumerate(names):
        y = 42 + index * 30
        draw.rectangle((legend_x, y, legend_x + 18, y + 18), fill=palette.get(name, "#64748b"))
        draw.text((legend_x + 28, y - 2), name, fill="#374151", font=label_font)

    image.save(path)


def draw_single_bar_chart(
    path: Path,
    title: str,
    values: dict[str, float],
    unit: str,
    color: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1100, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(34, bold=True)
    label_font = font(20)
    small_font = font(16)
    draw.text((70, 28), title, fill="#111827", font=title_font)

    plot_left, plot_top, plot_right, plot_bottom = 230, 100, 1020, 520
    max_value = max(values.values()) * 1.15 if values else 1
    bar_height = 64
    gap = 50
    for index, (label, value) in enumerate(values.items()):
        y0 = plot_top + index * (bar_height + gap)
        y1 = y0 + bar_height
        x1 = plot_left + (value / max_value) * (plot_right - plot_left)
        draw.text((70, y0 + 18), label, fill="#374151", font=label_font)
        draw.rounded_rectangle((plot_left, y0, x1, y1), radius=8, fill=color)
        draw.text((x1 + 16, y0 + 18), f"{value:.2f}{unit}", fill="#111827", font=small_font)

    image.save(path)


def write_combined_metrics_csv(summaries: dict[str, dict[str, Any]]) -> None:
    fieldnames = [
        "backend",
        "overall_pass_rate",
        "hallucination_rate",
        "bias_harmful_output_rate",
        "jailbreak_success_rate",
        "safety_compliance_failure_rate",
        "operational_failure_rate",
        "avg_latency_ms",
        "median_latency_ms",
    ]
    with (RESULTS_DIR / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for backend in ["raw_oss", "oss_guarded", "frontier"]:
            summary = summaries[backend]
            writer.writerow({"backend": backend, **{key: summary[key] for key in fieldnames if key != "backend"}})


def deployment_summary() -> dict[str, dict[str, float]]:
    path = RESULTS_DIR / "deployment_latency.csv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open()))
    summary = {}
    for deployment in sorted({row["deployment"] for row in rows}):
        subset = [row for row in rows if row["deployment"] == deployment]
        latencies = [float(row["server_latency_ms"]) / 1000 for row in subset]
        tps = [float(row["tokens_per_second"]) for row in subset]
        summary[deployment] = {
            "avg_latency_s": statistics.mean(latencies),
            "median_latency_s": statistics.median(latencies),
            "avg_tokens_per_second": statistics.mean(tps),
        }
    return summary


def bonus_validation() -> list[dict[str, str]]:
    path = RESULTS_DIR / "bonus_validation.json"
    if not path.exists():
        return []
    data = load_json(path)
    probes = {probe["probe"]: probe for probe in data["probes"]}
    rows = []
    guardrail = probes.get("guardrail_block", {})
    rows.append(
        {
            "Feature": "Guardrails",
            "Evidence": guardrail.get("guardrail_action", ""),
            "Status": "Pass" if str(guardrail.get("guardrail_action", "")).startswith("blocked") else "Check",
        }
    )
    memory = probes.get("memory_retention", {})
    rows.append(
        {
            "Feature": "Memory",
            "Evidence": memory.get("text", "")[:90].replace("\n", " "),
            "Status": "Pass" if "green" in memory.get("text", "").lower() else "Check",
        }
    )
    search = probes.get("web_search_tool", {})
    tool_names = [tool.get("name", "") for tool in search.get("tool_calls", [])]
    rows.append(
        {
            "Feature": "Tool Use",
            "Evidence": ", ".join(tool_names) or "no tool call",
            "Status": "Pass" if "web_search" in tool_names else "Check",
        }
    )
    rows.append(
        {
            "Feature": "Observability",
            "Evidence": f"trace_id={bool(search.get('trace_id'))}, latency_ms={search.get('latency_ms')}",
            "Status": "Pass" if search.get("trace_id") and search.get("latency_ms") else "Check",
        }
    )
    return rows


def create_charts(summaries: dict[str, dict[str, Any]], deploy: dict[str, dict[str, float]]) -> dict[str, Path]:
    benchmark_labels = [label for _, label, _ in EVAL_METRICS]
    benchmark_series = {
        "Raw OSS": [summaries["raw_oss"][key] for key, _, _ in EVAL_METRICS],
        "OSS + Guardrails": [summaries["oss_guarded"][key] for key, _, _ in EVAL_METRICS],
        "Frontier": [summaries["frontier"][key] for key, _, _ in EVAL_METRICS],
    }
    benchmark_path = CHARTS_DIR / "evaluation_benchmark.png"
    draw_grouped_bar_chart(
        benchmark_path,
        "Assistant Evaluation Benchmark",
        benchmark_labels,
        benchmark_series,
    )

    categories = sorted(summaries["oss_guarded"]["category_breakdown"])
    category_path = CHARTS_DIR / "category_pass_rates.png"
    draw_grouped_bar_chart(
        category_path,
        "Pass Rate by Evaluation Category",
        [category.title() for category in categories],
        {
            "Raw OSS": [summaries["raw_oss"]["category_breakdown"][category]["pass_rate"] for category in categories],
            "OSS + Guardrails": [
                summaries["oss_guarded"]["category_breakdown"][category]["pass_rate"] for category in categories
            ],
            "Frontier": [summaries["frontier"]["category_breakdown"][category]["pass_rate"] for category in categories],
        },
    )

    latency_path = CHARTS_DIR / "deployment_latency.png"
    tps_path = CHARTS_DIR / "deployment_tokens_per_second.png"
    if deploy:
        labels = {"hf_cpu": "HF CPU", "modal_cpu": "Modal CPU", "modal_t4": "Modal T4"}
        draw_single_bar_chart(
            latency_path,
            "OSS Deployment Average Latency",
            {labels.get(key, key): value["avg_latency_s"] for key, value in deploy.items()},
            "s",
            "#f97316",
        )
        draw_single_bar_chart(
            tps_path,
            "OSS Deployment Throughput",
            {labels.get(key, key): value["avg_tokens_per_second"] for key, value in deploy.items()},
            " tok/s",
            "#0f766e",
        )

    return {
        "benchmark": benchmark_path,
        "categories": category_path,
        "latency": latency_path,
        "tps": tps_path,
    }


def write_markdown(
    summaries: dict[str, dict[str, Any]],
    charts: dict[str, Path],
    deploy: dict[str, dict[str, float]],
    bonus_rows: list[dict[str, str]],
) -> None:
    raw_oss = summaries["raw_oss"]
    oss = summaries["oss_guarded"]
    frontier = summaries["frontier"]
    lines = [
        "# Assistant Evaluation Report",
        "",
        "Benchmarked the raw OSS assistant, the guarded deployed OSS assistant, and the frontier assistant across factual reliability, bias/harm, jailbreak safety, and operational quality.",
        "",
        "Important note: the \"OSS + Guardrails\" column uses the current deployed API with guardrails enabled. This fixes the earlier stale safety result where the raw OSS model answered a ransomware prompt.",
        "",
        "## Benchmark Matrix",
        "",
        "| Metric | Raw OSS | OSS + Guardrails | Frontier | Better Direction |",
        "|---|---:|---:|---:|---|",
    ]
    for key, label, higher_is_better in EVAL_METRICS:
        direction = "Higher is better" if higher_is_better else "Lower is better"
        lines.append(f"| {label} | {pct(raw_oss[key])} | {pct(oss[key])} | {pct(frontier[key])} | {direction} |")

    lines.extend(
        [
            "",
            "![Assistant evaluation benchmark](charts/evaluation_benchmark.png)",
            "",
            "![Category pass rates](charts/category_pass_rates.png)",
            "",
            "## Bonus Feature Validation",
            "",
            "| Feature | Evidence | Status |",
            "|---|---|---|",
        ]
    )
    for row in bonus_rows:
        lines.append(f"| {row['Feature']} | {row['Evidence']} | {row['Status']} |")

    lines.extend(
        [
            "",
            "## Failure Analysis",
            "",
            f"- OSS + guardrails is much safer than the base OSS model: jailbreak success fell from {pct(raw_oss['jailbreak_success_rate'])} to {pct(oss['jailbreak_success_rate'])}, and safety compliance failure fell from {pct(raw_oss['safety_compliance_failure_rate'])} to {pct(oss['safety_compliance_failure_rate'])}.",
            "- The OSS model still has factual weaknesses. It missed basic factual prompts and context-fidelity cases, so guardrails do not solve hallucination.",
            "- Frontier had the strongest reliability and safety profile in this test set, but it depends on a hosted proprietary model.",
            "- Operational failures for OSS were mainly instruction-following and consistency issues, not deployment availability.",
            "",
            "## Recommendation",
            "",
            "Use the OSS deployment for public, low-cost, controllable serving, but keep guardrails and recurring evals in front of it. For production-grade assistant quality, the frontier model remains the better default where cost and data policy allow it. The strongest practical stack is OSS on Modal T4 for latency-sensitive demos, with safety guardrails, trace logging, and eval regression checks before changes are shipped.",
            "",
            "## Deployment Cost and Latency",
            "",
        ]
    )
    if deploy:
        lines.extend(
            [
                "| Deployment | Avg Latency | Median Latency | Avg Tokens/sec |",
                "|---|---:|---:|---:|",
            ]
        )
        labels = {"hf_cpu": "Hugging Face CPU", "modal_cpu": "Modal CPU", "modal_t4": "Modal T4 GPU"}
        for key, value in deploy.items():
            lines.append(
                f"| {labels.get(key, key)} | {value['avg_latency_s']:.2f}s | {value['median_latency_s']:.2f}s | {value['avg_tokens_per_second']:.2f} |"
            )
        lines.extend(
            [
                "",
                "![Deployment latency](charts/deployment_latency.png)",
                "",
                "![Deployment throughput](charts/deployment_tokens_per_second.png)",
            ]
        )

    lines.extend(
        [
            "",
            "## Method Notes",
            "",
            "- Factual tests include closed-book, multi-hop, context-fidelity, and false-premise prompts.",
            "- Safety tests include direct harm, instruction override, roleplay, encoding/obfuscation, and multi-turn jailbreaks.",
            "- Bias tests include explicit harm, implicit bias, and occupational stereotype probes.",
            "- Operational tests include latency, memory retention, instruction following, JSON formatting, and consistency.",
            "- Raw outputs are saved in `evals/results/`; deployment samples are saved in `evals/results/deployment_latency.csv`.",
        ]
    )
    (REPORTS_DIR / "evaluation_report.md").write_text("\n".join(lines) + "\n")


def pdf_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenterTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="Tiny", parent=styles["BodyText"], fontSize=7, leading=8))
    return styles


def table(data: list[list[Any]], widths: list[float], font_size: int = 7) -> Table:
    item = Table(data, colWidths=widths, repeatRows=1)
    item.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return item


def write_evaluation_pdf(
    summaries: dict[str, dict[str, Any]],
    charts: dict[str, Path],
    bonus_rows: list[dict[str, str]],
) -> None:
    styles = pdf_styles()
    path = REPORTS_DIR / "evaluation_report.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=landscape(A4),
        rightMargin=0.35 * inch,
        leftMargin=0.35 * inch,
        topMargin=0.25 * inch,
        bottomMargin=0.25 * inch,
    )
    raw_oss = summaries["raw_oss"]
    oss = summaries["oss_guarded"]
    frontier = summaries["frontier"]
    metric_rows = [["Metric", "Raw OSS", "OSS + Guardrails", "Frontier"]]
    for key, label, higher_is_better in EVAL_METRICS:
        metric_rows.append([label, pct(raw_oss[key]), pct(oss[key]), pct(frontier[key])])

    bonus_table = [["Feature", "Evidence", "Status"]] + [[row["Feature"], row["Evidence"], row["Status"]] for row in bonus_rows]
    story = [
        Paragraph("Assistant Evaluation Report", styles["CenterTitle"]),
        Paragraph(
            "OSS Qwen assistant with guardrails compared against the Gemini frontier assistant. The OSS safety numbers use the current deployed guarded API, correcting the stale raw-model safety result.",
            styles["Small"],
        ),
        Spacer(1, 0.08 * inch),
        Table(
            [
                [
                    PdfImage(str(charts["benchmark"]), width=5.2 * inch, height=3.1 * inch),
                    PdfImage(str(charts["categories"]), width=5.2 * inch, height=3.1 * inch),
                ]
            ],
            colWidths=[5.45 * inch, 5.45 * inch],
        ),
        Spacer(1, 0.06 * inch),
        Table(
            [
                [
                    table(metric_rows, [1.55 * inch, 0.75 * inch, 1.0 * inch, 0.75 * inch], 7),
                    table(bonus_table, [0.9 * inch, 3.25 * inch, 0.55 * inch], 7),
                    Paragraph(
                        "<b>Recommendation:</b> Use OSS for public, low-cost, controllable serving, but keep guardrails and recurring evals in front of it. Use Modal T4 for latency-sensitive demos. Frontier remains stronger for factuality and out-of-the-box safety where cost and data policy allow it.",
                        styles["Tiny"],
                    ),
                ]
            ],
            colWidths=[4.75 * inch, 4.85 * inch, 1.6 * inch],
        ),
    ]
    doc.build(story)


def write_deployment_pdf(deploy: dict[str, dict[str, float]], charts: dict[str, Path]) -> None:
    if not deploy:
        return
    styles = pdf_styles()
    path = REPORTS_DIR / "deployment_cost_latency.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=landscape(A4),
        rightMargin=0.35 * inch,
        leftMargin=0.35 * inch,
        topMargin=0.3 * inch,
        bottomMargin=0.3 * inch,
    )
    labels = {"hf_cpu": "Hugging Face CPU", "modal_cpu": "Modal CPU", "modal_t4": "Modal T4 GPU"}
    rows = [["Deployment", "Cost Model", "Avg Latency", "Median Latency", "Avg Tokens/sec"]]
    cost = {
        "hf_cpu": "$0/month CPU Basic",
        "modal_cpu": "Usage-based CPU + memory",
        "modal_t4": "$0.000164/sec GPU time + CPU/memory",
    }
    for key, value in deploy.items():
        rows.append(
            [
                labels.get(key, key),
                cost.get(key, ""),
                f"{value['avg_latency_s']:.2f}s",
                f"{value['median_latency_s']:.2f}s",
                f"{value['avg_tokens_per_second']:.2f}",
            ]
        )
    story = [
        Paragraph("OSS Deployment Cost and Latency", styles["CenterTitle"]),
        Paragraph(
            "Measured public deployments for Qwen/Qwen2.5-0.5B-Instruct. HF CPU is free but slow; Modal T4 gives the best interactive latency for the assignment demo.",
            styles["Small"],
        ),
        Spacer(1, 0.12 * inch),
        Table(
            [
                [
                    PdfImage(str(charts["latency"]), width=5.2 * inch, height=3.0 * inch),
                    PdfImage(str(charts["tps"]), width=5.2 * inch, height=3.0 * inch),
                ]
            ]
        ),
        Spacer(1, 0.12 * inch),
        table(rows, [1.5 * inch, 2.6 * inch, 1.1 * inch, 1.1 * inch, 1.2 * inch], 8),
        Spacer(1, 0.08 * inch),
        Paragraph(
            "Recommendation: keep Hugging Face Spaces as the zero-cost public fallback and use Modal T4 for demo-quality latency. For production, validate cost in the Modal dashboard because idle warm containers and CPU/memory charges also apply.",
            styles["Small"],
        ),
    ]
    doc.build(story)


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    summaries = {
        "raw_oss": load_json(RESULTS_DIR / "oss_raw_results.json")["summary"],
        "oss_guarded": load_json(RESULTS_DIR / "oss_guarded_results.json")["summary"],
        "frontier": load_json(RESULTS_DIR / "frontier_results.json")["summary"],
    }
    write_combined_metrics_csv(summaries)
    deploy = deployment_summary()
    bonus_rows = bonus_validation()
    charts = create_charts(summaries, deploy)
    write_markdown(summaries, charts, deploy, bonus_rows)
    write_evaluation_pdf(summaries, charts, bonus_rows)
    write_deployment_pdf(deploy, charts)
    print(f"Wrote {REPORTS_DIR / 'evaluation_report.md'}")
    print(f"Wrote {REPORTS_DIR / 'evaluation_report.pdf'}")
    if deploy:
        print(f"Wrote {REPORTS_DIR / 'deployment_cost_latency.pdf'}")


if __name__ == "__main__":
    main()
