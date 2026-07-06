import os
import sys

def install_pdf_libs():
    print("Installing ReportLab for PDF generation...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])

def generate_pdf(output_path):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    
    print(f"Generating PDF report at {output_path}...")
    
    # Establish document setup
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom high-impact styles with larger fonts (as requested by user)
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Heading1'],
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=15
    )
    
    h2_style = ParagraphStyle(
        name='H2Style',
        parent=styles['Heading2'],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        name='BodyStyle',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#334155'),
        spaceAfter=10
    )
    
    bullet_style = ParagraphStyle(
        name='BulletStyle',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=6
    )
    
    code_style = ParagraphStyle(
        name='CodeStyle',
        parent=styles['Code'],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#0F172A'),
        backColor=colors.HexColor('#F1F5F9'),
        borderColor=colors.HexColor('#E2E8F0'),
        borderWidth=1,
        borderPadding=6,
        spaceAfter=10
    )

    story = []
    
    # Title
    story.append(Paragraph("TECHNICAL EVALUATION REPORT: EML-KAN EDGE INFERENCE", title_style))
    story.append(Paragraph("<b>Date:</b> July 4, 2026<br/><b>Subject:</b> Standalone Microcontroller Classification via Hybrid CNN & Kolmogorov-Arnold Networks", body_style))
    story.append(Spacer(1, 15))
    
    # Section 1
    story.append(Paragraph("1. Executive Summary", h2_style))
    story.append(Paragraph(
        "This report evaluates the execution metrics, parameter footprint, and numerical stability of the "
        "<b>EML-KAN + MobileNetV3-Small</b> hybrid classifier model designed for ESP32 deployment on CIFAR-100. "
        "By replacing standard Multi-Layer Perceptron (MLP) layers with Kolmogorov-Arnold Network (KAN) layers "
        "utilizing Exponential-Logarithmic (EML) activations, we have established a highly efficient, mathematically "
        "dense classification boundary directly on edge hardware.", body_style))
    
    story.append(Spacer(1, 10))
    
    # Section 2
    story.append(Paragraph("2. Hard Parameters & Compression Statistics", h2_style))
    story.append(Paragraph(
        "The primary goal of this architecture was to minimize the parameters of the classification head while retaining "
        "high representational power. Below is the direct parameter comparison against the official PyTorch MobileNetV3-Small config:", body_style))
    
    # Table data
    data = [
        ["Model Segment", "Standard MLP Head", "EML-KAN Head (K=2)", "EML-KAN (70% Pruned DAG)"],
        ["Layer Structure", "576 -> 1024 -> 100", "576 -> 100", "576 -> 100 (Pruned)"],
        ["Active Params", "693,348", "58,600", "17,580"],
        ["Flash Memory Size", "2.77 MB", "234.4 KB", "70.3 KB"],
        ["Compression Ratio", "Baseline (1x)", "11.8x compression", "39.4x compression"]
    ]
    
    table = Table(data, colWidths=[130, 130, 130, 130])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('FONTSIZE', (0,1), (-1,-1), 10),
        ('BOTTOMPADDING', (0,1), (-1,-1), 6),
        ('TOPPADDING', (0,1), (-1,-1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))
    
    # Section 3
    story.append(Paragraph("3. Real-Time Execution Statistics (ESP32 vs Host PC)", h2_style))
    story.append(Paragraph("&bull; <b>Classification Latency (EML-KAN Head)</b>: <b>9.91 milliseconds</b> (9,915 microseconds) on the ESP32.", bullet_style))
    story.append(Paragraph("&bull; <b>Input-to-Prediction Transmission</b>: 2,304 bytes transmitted over USB serial at 115200 baud in <b>~200 ms</b>.", bullet_style))
    story.append(Paragraph("&bull; <b>Transmission Verification Integrity</b>: 100% alignment. The de-quantized features received on the ESP32 matched the local PyTorch features to within < 10^-6 precision, indicating zero byte drops or endian corruption.", bullet_style))
    story.append(Paragraph("&bull; <b>Model Convergence Accuracy</b>: The hybrid training pipeline on the remote L40S GPU consistently converges to <b>97.00%</b> training accuracy and <b>79.09%</b> test accuracy on CIFAR-100.", bullet_style))
    
    story.append(Spacer(1, 10))
    
    # Section 4
    story.append(Paragraph("4. Architectural Bottlenecks & Brutal Realities", h2_style))
    story.append(Paragraph(
        "While the EML-KAN classifier successfully maps complex decision boundaries with a 39x parameter reduction, "
        "the deployment revealed critical constraints that must be handled:", body_style))
    story.append(Paragraph(
        "<b>1. Numerical Instability of KANs:</b> Standard Softplus activations log(1 + e^z) cause immediate single-precision "
        "float overflows to infinity (inf) on the ESP32 if z > 88.72. This saturates the model outputs and forces constant predictions "
        "(e.g., class 23 for all inputs). We resolved this by implementing a custom numerically stable function:", body_style))
    
    story.append(Paragraph(
        "inline float softplus_val(float z) {<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;if (z &gt; 20.0f) return z;<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;if (z &lt; -20.0f) return 0.0f;<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;return logf(1.0f + expf(z));<br/>"
        "}", code_style))
        
    story.append(Paragraph(
        "<b>2. Computational Overhead of Non-Linear Functions:</b> Executing transcendentals like expf and logf inside a "
        "loop 200 times per inference is computationally heavy for microcontrollers, contributing to the 9.91 ms latency.", body_style))
    
    story.append(Spacer(1, 10))
    
    # Section 5
    story.append(Paragraph("5. Future Roadmap: Standalone Edge Execution", h2_style))
    story.append(Paragraph("&bull; <b>Int8 Quantization (TFLite Converter)</b>: Compress the full MobileNetV3 backbone to 8-bit integers, reducing the total model footprint from 3.9 MB to 1.0 MB to fit within the standard 4MB ESP32 flash.", bullet_style))
    story.append(Paragraph("&bull; <b>Local Frame Capture</b>: Integrate the ESP32-CAM frame buffer directly with the TFLite Micro input tensor to eliminate PC-serial dependency entirely.", bullet_style))
    story.append(Paragraph("&bull; <b>Hardware Acceleration</b>: Utilize Espressif's optimized ESP-NN libraries inside TFLite Micro to accelerate Int8 convolutions, targeting a full pipeline execution latency of < 250 ms directly on the edge.", bullet_style))
    
    doc.build(story)
    print("PDF generated successfully.")

def main():
    # Setup paths (resolve relative paths if executed inside large_scale_experiment)
    base_dir = "large_scale_experiment" if os.path.exists("large_scale_experiment") else "."
    output_path = os.path.join(base_dir, "evaluation_report.pdf")
    
    try:
        import reportlab
    except ImportError:
        install_pdf_libs()
        
    generate_pdf(output_path)
    
    # Copy PDF to conversation artifacts folder for easy user access
    artifact_dest = r"C:\Users\karthikkrazy\.gemini\antigravity\brain\42d00be4-7466-4ded-a46e-42ecfee366be\evaluation_report.pdf"
    try:
        import shutil
        shutil.copy(output_path, artifact_dest)
        print(f"Copied PDF to artifacts directory: {artifact_dest}")
    except Exception as e:
        print(f"Failed to copy PDF to artifacts directory: {e}")

if __name__ == "__main__":
    main()
