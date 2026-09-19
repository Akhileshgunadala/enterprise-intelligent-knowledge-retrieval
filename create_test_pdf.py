from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

input_file = "app/test_new_document.txt"
output_file = "University_Course_Registration_Policy.pdf"

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

doc = SimpleDocTemplate(
    output_file,
    pagesize=A4,
    rightMargin=50,
    leftMargin=50,
    topMargin=50,
    bottomMargin=50
)

styles = getSampleStyleSheet()

title_style = styles["Title"]
heading_style = styles["Heading2"]
body_style = styles["BodyText"]

story = []

for line in text.splitlines():

    line = line.strip()

    if not line:
        story.append(Spacer(1, 8))
        continue

    if line.isupper():
        story.append(
            Paragraph(line, heading_style)
        )
    else:
        story.append(
            Paragraph(line, body_style)
        )

    story.append(Spacer(1, 5))

doc.build(story)

print(f"PDF created successfully: {output_file}")