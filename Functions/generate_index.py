from .LIBRARIES_FUNCTIONS import *

# To generate content index for notebook
def generate_index(file="Restyle.ipynb", title="Index"):
    with open(file, "r", encoding="utf-8") as f:
        nb = json.load(f)

    headers = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            for line in cell["source"]:
                m = re.match(r'^(#+)\s+(.*)', line)
                if m:
                    level = len(m.group(1))
                    text = m.group(2).strip()

                    anchor = re.sub(r'[^a-zA-Z0-9 -]', '', text)
                    anchor = anchor.replace(" ", "-")

                    headers.append((level, text, anchor))

    # HTML style
    md = f"""
<h1 style="color:black; font-size: 38px; font-weight: 700; margin-bottom: 5px;">
    {title}
</h1>

<hr style="border: 1px solid #000;">

<p style="font-size: 18px; color:black; margin-top: 10px;">

</p>
"""

    for level, text, anchor in headers:
        indent = "&nbsp;" * (level - 1) * 6
        size = 20 if level == 1 else 17
        weight = "700" if level == 1 else "500"
        bullet = "•" if level == 1 else "◦"

        md += (
            f'{indent}<span style="font-size:{size}px; color:black; font-weight:{weight};">'
            f'{bullet} <a href="#{anchor}" style="color:black; text-decoration:none;">{text}</a>'
            f'</span><br>\n'
        )

    md += '<br>\n'
    md += '<hr style="border: 1px solid #000;">\n'
    # md += '<hr style="border: 1px solid #000;">\n'
    md += '<br>\n'
    # md += '<br>\n'

    display(Markdown(md))