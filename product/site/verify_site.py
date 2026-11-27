"""Verify site links, bilingual routes, benchmark data and static bundle contents."""
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from xml.etree.ElementTree import parse

root = Path(__file__).resolve().parents[2]
dist = root / "main/dist"
pages = sorted(dist.rglob("*.html"))

class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.lang = None
        self.h1s = 0

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "html":
            self.lang = values.get("lang")
        elif tag == "h1":
            self.h1s += 1
        for attr in ("href", "src"):
            if attr in values:
                self.links.append(values[attr])

links = 0
for file in pages:
    page = Page()
    page.feed(file.read_text("utf-8"))
    location = "/" + file.parent.relative_to(dist).as_posix().strip(".") + "/"
    location = "/" if location == "//" else location
    language = "en" if location.startswith("/en/") else "ja"
    assert (page.lang, page.h1s) == (language,1), file
    alternate = location.removeprefix("/en") if language == "en" else "/en" + location
    assert alternate in page.links, (file,alternate)
    for href in page.links:
        parsed = urlsplit(urljoin("https://review.invalid"+location,href))
        if parsed.netloc != "review.invalid":
            continue
        path = dist / parsed.path.lstrip("/")
        if parsed.path.endswith("/"):
            path /= "index.html"
        assert path.resolve().is_relative_to(dist.resolve()) and path.is_file(), (file,href)
        links += 1

for file in dist.rglob("*"):
    if file.is_file():
        assert file.suffix in {".html",".css",".js",".jpg",".png",".svg",".csv",".json",".webp",".ico"}, file
        if file.suffix == ".svg":
            parse(file)
        if file.suffix == ".json":
            json.loads(file.read_text("utf-8-sig"))
details = json.loads((dist/"data/details.json").read_text("utf-8"))
public = json.loads((dist/"data/evaluation.json").read_text("utf-8"))
assert len([r for r in public["models"] if r["scope"] == "local_control"]) == 1
for row in public["models"]:
    if row["scope"] == "official_validation":
        assert abs(row["ap50_95"] - 100 * details["models"][row["id"]]["scores"]["ap"]) < 0.0001
assert len(details["timings"]) == len([r for r in public["runtime"] if r["status"] == "completed"])
print(f"Verified {len(pages)} pages, {links} internal references, {len(public['charts'])} plots and {len(details['timings'])} runtime configurations.")
