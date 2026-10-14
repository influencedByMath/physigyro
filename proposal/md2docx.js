// Convert proposal.md (simple Markdown subset) to a one-page A4 proposal.docx.
// Usage (needs Node.js): npm install docx@9.5.1   then   node proposal/md2docx.js proposal/proposal.md proposal/proposal.docx
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, BorderStyle, ShadingType, LevelFormat,
} = require("docx");

const [, , inPath, outPath] = process.argv;
const lines = fs.readFileSync(inPath, "utf8").split(/\r?\n/);

const FONT = "Calibri", SIZE = 20; // half-points: 10 pt
const PAGE_W = 11906, MARGIN = 850; // A4 width, 0.59 in margins
const CONTENT_W = PAGE_W - 2 * MARGIN;

// Inline parser: **bold**, *italic*, $M_s$ math (letter + subscript only)
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\$[^$]+\$)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(...runs(tok.slice(2, -2), { ...base, bold: true }));
    else if (tok.startsWith("$")) {
      const [main, sub] = tok.slice(1, -1).split("_");
      out.push(new TextRun({ text: main, italics: true, ...base }));
      if (sub) out.push(new TextRun({ text: sub, subScript: true, ...base }));
    } else out.push(new TextRun({ text: tok.slice(1, -1), ...base, italics: true }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const children = [];
const para = (text, opts = {}) =>
  children.push(new Paragraph({ children: runs(text), spacing: { after: 70, line: 240 }, ...opts }));

let table = null;
const border = { style: BorderStyle.SINGLE, size: 4, color: "BBBBBB" };
const cellBorders = { top: border, bottom: border, left: border, right: border };
const COLS = [1900, CONTENT_W - 1900];

function flushTable() {
  if (!table) return;
  const rows = table.filter((r) => !/^\|\s*-/.test(r)).map((r, i) => {
    const cells = r.split("|").slice(1, -1).map((c) => c.trim());
    return new TableRow({
      children: cells.map((c, j) => new TableCell({
        width: { size: COLS[j], type: WidthType.DXA },
        borders: cellBorders,
        margins: { top: 20, bottom: 20, left: 80, right: 80 },
        shading: i === 0 ? { type: ShadingType.CLEAR, color: "auto", fill: "E8EEF4" } : undefined,
        children: [new Paragraph({ children: runs(c, i === 0 ? { bold: true } : {}), spacing: { after: 0, line: 240 } })],
      })),
    });
  });
  children.push(new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: COLS, rows }));
  children.push(new Paragraph({ children: [], spacing: { after: 60 } }));
  table = null;
}

for (const raw of lines) {
  const line = raw.trimEnd();
  if (line.startsWith("|")) { (table = table || []).push(line); continue; }
  flushTable();
  if (!line.trim()) continue;
  if (line.startsWith("# ")) continue; // "Thesis proposal" label goes into the subtitle line
  if (line.startsWith("## ")) {
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
      children: [new TextRun({ text: "Thesis proposal", size: 18, color: "555555", allCaps: true })] }));
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
      children: [new TextRun({ text: line.slice(3), bold: true, size: 28 })] }));
    continue;
  }
  if (line.startsWith("**Student:**")) { para(line, { alignment: AlignmentType.CENTER, spacing: { after: 120 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "999999", space: 4 } } }); continue; }
  let m;
  if ((m = line.match(/^(\d+)\.\s+(.*)$/))) {
    children.push(new Paragraph({ numbering: { reference: "num", level: 0 }, children: runs(m[2]), spacing: { after: 30, line: 240 } }));
    continue;
  }
  if ((m = line.match(/^-\s+(.*)$/))) {
    children.push(new Paragraph({ numbering: { reference: "bul", level: 0 }, children: runs(m[1]), spacing: { after: 30, line: 240 } }));
    continue;
  }
  para(line);
}
flushTable();

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: SIZE } } } },
  numbering: { config: [
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 360, hanging: 260 } } } }] },
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 360, hanging: 260 } } } }] },
  ] },
  sections: [{ properties: { page: { size: { width: PAGE_W, height: 16838 },
    margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } }, children }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(outPath, b); console.log("wrote", outPath, b.length, "bytes"); });
