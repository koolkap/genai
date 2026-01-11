import pymupdf

def extract_without_header_footer(pdf_path, header_percent=0.10, footer_percent=0.10):
    """
    Extracts text from PDF while excluding headers and footers.
    
    header_percent and footer_percent define how much of the page height to ignore.
    """
    doc = pymupdf.open(pdf_path)
    full_text = ""

    for page in doc:
        page_height = page.rect.height

        # define cutoff regions
        header_cut = page_height * header_percent
        footer_cut = page_height * (1 - footer_percent)

        # extract text blocks
        blocks = page.get_text("blocks")

        for block in blocks:
            x0, y0, x1, y1, text, block_no = block[:6]

            # keep only blocks that are not in header/footer regions
            if header_cut < y0 < footer_cut:
                full_text += text.strip() + "\n"

    doc.close()
    return full_text


# Example usage
pdf_path = "The Almanack of Naval Ravikant PDF.pdf"
clean_text = extract_without_header_footer(pdf_path)
print(clean_text)
