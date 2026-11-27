"""Optional additional products live on the same Site and in the same navigation."""
import re
from html import escape

def entries(config):
    result = {}
    for item in config.get("additional_products", []):
        if not re.fullmatch(r"[a-z][a-z0-9-]*", item["slug"]):
            raise ValueError("Product slug must be lowercase ASCII letters, digits and hyphens.")
        page = "catalog/" + item["slug"]
        if page in result or item["kind"] not in {"model","software","learning"}:
            raise ValueError("Duplicate product slug or unknown kind.")
        for field in ("name","summary","description"):
            if not all(isinstance(item[field].get(lang), str) and item[field][lang].strip() for lang in ("en","ja")):
                raise ValueError(f"Missing bilingual product field: {field}")
        result[page] = item
    return result

def localized(item, field, english):
    return escape(item[field]["en" if english else "ja"])
