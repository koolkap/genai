import pymupdf

pdf_path = "The Almanack of Naval Ravikant PDF.pdf"

doc = pymupdf.open(pdf_path)
text = ""

for page in doc:
    text += page.get_text()

doc.close()

print(text)
