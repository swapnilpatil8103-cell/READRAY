"""Generate assets/sample_paper.pdf — a short, clearly FICTIONAL paper used for demos and tests.

Run:  python scripts/make_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

OUT = Path(__file__).resolve().parents[1] / "assets" / "sample_paper.pdf"

CSS = """
body { font-family: serif; font-size: 10.5pt; line-height: 1.35; }
h1 { font-size: 17pt; text-align: center; margin-bottom: 4pt; }
.authors { text-align: center; font-size: 11pt; margin-bottom: 2pt; }
.note { text-align: center; font-size: 8.5pt; color: #8a3b12; margin-bottom: 10pt; }
h2 { font-size: 12.5pt; font-weight: bold; margin-top: 12pt; margin-bottom: 4pt; }
p { text-align: justify; margin-top: 0; margin-bottom: 6pt; }
.eq { text-align: center; font-style: italic; margin: 6pt 0; }
.caption { font-size: 9pt; margin-bottom: 8pt; }
table { border-collapse: collapse; margin: 4pt auto; font-size: 9.5pt; }
td, th { border: 0.5pt solid #444; padding: 2pt 6pt; text-align: center; }
.small { font-size: 9pt; }
"""

HTML = """
<h1>Interest Rate Announcements and Small-Firm Credit: A Fictional Event Study</h1>
<p class="authors">A. Example and B. Sample</p>
<p class="note">SAMPLE DOCUMENT FOR DEMONSTRATING READRAY. The study, data and results below are invented and are not real research findings.</p>

<p><b>Abstract</b> We study how unexpected policy-rate announcements relate to credit spreads charged to small firms in a
fictional economy. Using a panel of 1,200 invented loan contracts observed between 2015 and 2022, we estimate an event-study
regression around 32 announcement dates. A one-percentage-point surprise increase is associated with a 0.41 percentage-point
rise in small-firm spreads within one month, compared with 0.18 points for large firms. The association is weaker when banks
hold more liquid assets. Because announcement timing may coincide with other news, we interpret the estimates as
associations rather than clean causal effects.</p>

<h2>1 Introduction</h2>
<p>Central banks change interest rates to influence borrowing costs across the economy, but the pass-through may differ
by borrower size. Small firms rely more on bank credit and have fewer outside options, so lenders may adjust their terms more
quickly. This paper asks whether surprise policy announcements are followed by larger spread changes for small firms than for
large firms, and whether bank liquidity dampens that difference.</p>
<p>Our contribution is a transparent event-study design applied to contract-level spreads. We build on the idea of a bank
lending channel, in which monetary policy affects the supply of loans through bank balance sheets.</p>

<h2>2 Data</h2>
<p>The fictional dataset contains 1,200 loan contracts issued by 15 banks to 640 firms. Firms with fewer than 50 employees
are classified as small. For each contract we observe the spread over the policy rate, maturity, collateral, and the issuing
bank's ratio of liquid assets to total assets. Announcement surprises are measured as the change in a one-month market
rate in a 30-minute window around each announcement.</p>
<p>Table 1 reports summary statistics. Small firms pay higher average spreads and borrow at shorter maturities.</p>

<p class="caption"><b>Table 1:</b> Summary statistics by firm size (fictional data).</p>
<table>
<tr><th>Group</th><th>Contracts</th><th>Mean spread (pp)</th><th>Mean maturity (months)</th><th>Collateralised (%)</th></tr>
<tr><td>Small firms</td><td>730</td><td>3.12</td><td>18</td><td>71</td></tr>
<tr><td>Large firms</td><td>470</td><td>1.64</td><td>41</td><td>38</td></tr>
<tr><td>All</td><td>1200</td><td>2.54</td><td>27</td><td>58</td></tr>
</table>

<h2>3 Methodology</h2>
<p>We estimate the following regression for contract <i>i</i> issued by bank <i>b</i> in month <i>t</i>:</p>
<p class="eq">Δs<sub>i,t</sub> = α + β·S<sub>t</sub> + δ·(S<sub>t</sub> × Small<sub>i</sub>) + θ·(S<sub>t</sub> × Small<sub>i</sub> × Liq<sub>b</sub>) + γ′X<sub>i,t</sub> + ε<sub>i,t</sub>     (1)</p>
<p>where Δs is the change in the loan spread relative to the bank's previous comparable contract, S is the announcement
surprise in percentage points, Small is an indicator for small firms, Liq is the bank's liquid-asset ratio, and X contains
maturity and collateral controls. The coefficient δ captures the additional spread response of small firms. Standard errors
are clustered by bank. Identification relies on the assumption that, within the 30-minute window, the surprise is unrelated
to other information that affects loan pricing.</p>

<h2>4 Results</h2>
<p>Table 2 presents the main estimates. The coefficient on the interaction term is positive and statistically significant:
small-firm spreads rise by 0.41 percentage points after a one-point surprise (β + δ), compared with 0.18 points for large firms.
The triple interaction θ is negative, indicating that the small-firm premium shrinks when banks are more liquid.</p>

<p class="caption"><b>Table 2:</b> Event-study estimates of equation (1) (fictional; standard errors in parentheses).</p>
<table>
<tr><th>Coefficient</th><th>Estimate</th><th>Std. error</th></tr>
<tr><td>β (surprise)</td><td>0.18</td><td>(0.05)</td></tr>
<tr><td>δ (surprise × small)</td><td>0.23</td><td>(0.07)</td></tr>
<tr><td>θ (surprise × small × liquidity)</td><td>-0.52</td><td>(0.21)</td></tr>
<tr><td>Observations</td><td>1200</td><td></td></tr>
</table>

<p>Figure 1 plots the average spread change in the months around announcements, separately for small and large firms.</p>
<img src="fig1.png" width="380"/>
<p class="caption"><b>Figure 1:</b> Average change in loan spreads (percentage points) by month relative to the announcement,
small versus large firms (fictional data).</p>

<h2>5 Limitations</h2>
<p>Several limitations apply. First, other news released on announcement days could affect spreads, so the estimates may
not be purely causal. Second, we observe only contracts that were approved; if banks ration credit to small firms instead of
raising prices, spreads understate the tightening. Third, the sample covers a single fictional country and period, which
limits external validity.</p>

<h2>6 Conclusion</h2>
<p>In this fictional setting, surprise rate increases are associated with larger spread increases for small firms, and
bank liquidity is associated with a smaller gap. Future work could examine loan volumes and rejection rates to capture
credit rationing.</p>

<h2>References</h2>
<p class="small">Example, A. (2020). A fictional review of the bank lending channel. <i>Journal of Illustrative Economics</i>, 1(1), 1–10.</p>
<p class="small">Sample, B. (2021). Invented evidence on loan pricing. <i>Demonstration Finance Review</i>, 2(3), 11–20.</p>
"""


def _figure_png() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=420, height=230)
    months = [-2, -1, 0, 1, 2, 3]
    small = [0.02, 0.01, 0.15, 0.41, 0.37, 0.33]
    large = [0.01, 0.00, 0.07, 0.18, 0.16, 0.15]
    x0, y0, w, h = 50, 30, 340, 160
    page.draw_line((x0, y0 + h), (x0 + w, y0 + h))
    page.draw_line((x0, y0), (x0, y0 + h))
    for tick in (0.0, 0.1, 0.2, 0.3, 0.4):
        y = y0 + h - tick / 0.45 * h
        page.insert_text((x0 - 28, y + 3), f"{tick:.1f}", fontsize=7)
    group = w / len(months)
    for i, m in enumerate(months):
        gx = x0 + i * group + 8
        for j, (vals, color) in enumerate(((small, (0.23, 0.36, 0.57)), (large, (0.78, 0.5, 0.23)))):
            bh = vals[i] / 0.45 * h
            page.draw_rect(fitz.Rect(gx + j * 18, y0 + h - bh, gx + j * 18 + 16, y0 + h), color=None, fill=color)
        page.insert_text((gx + 8, y0 + h + 12), str(m), fontsize=7)
    page.insert_text((x0 + w / 2 - 70, y0 + h + 26), "Months relative to announcement", fontsize=8)
    page.insert_text((x0 + 10, y0 - 10), "Spread change (pp):", fontsize=8)
    page.draw_rect(fitz.Rect(300, 18, 308, 26), color=None, fill=(0.23, 0.36, 0.57))
    page.insert_text((312, 25), "Small firms", fontsize=7)
    page.draw_rect(fitz.Rect(300, 30, 308, 38), color=None, fill=(0.78, 0.5, 0.23))
    page.insert_text((312, 37), "Large firms", fontsize=7)
    return page.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")


def build(out: Path = OUT) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    archive = fitz.Archive()
    archive.add(_figure_png(), "fig1.png")
    story = fitz.Story(html=HTML, user_css=CSS, archive=archive)
    mediabox = fitz.paper_rect("letter")
    where = mediabox + (72, 66, -72, -66)
    import io

    buffer = io.BytesIO()
    writer = fitz.DocumentWriter(buffer)
    more = 1
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    doc = fitz.open(stream=buffer.getvalue(), filetype="pdf")
    doc.set_metadata({"title": "Interest Rate Announcements and Small-Firm Credit: A Fictional Event Study",
                      "author": "A. Example and B. Sample", "subject": "Readray sample document (fictional)"})
    doc.save(out, garbage=3, deflate=True)
    doc.close()
    return out


if __name__ == "__main__":
    print(f"Wrote {build()}")
