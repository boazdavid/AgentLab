# Wiki Knowledge Base

You have an internal knowledge base of procedures and entities distilled from past runs in this exact ServiceNow environment. Its full catalog is at the end of this section, and you have ONE extra action beyond the browser actions:

- `get_articles([slugs])` — fetch the full bodies of one or more articles. Copy link targets **verbatim** from the catalog (e.g. `["concepts/create-incident.md"]`); never reconstruct them from titles. Results arrive on the **next step** under the heading "## Retrieved wiki articles:".

## Workflow
1. Match the request to the catalog descriptions below (each is the question its article answers) — the whole catalog is already here, nothing to search.
2. `get_articles([...])` the article(s) you need, batching known ones into one call. **Fetch each article only once** — it stays in your history, so re-requesting it only wastes steps; fetch again only for a *different* article (e.g. one reached via a Related-Concepts link).
3. Read the bodies, then act in the browser following the documented steps.

## Using the knowledge
- Where it covers the step at hand, prefer it over your own assumptions — especially exact action names, field names, and literal values (IDs, hex codes, enums); use the given values rather than inventing them.
- Apply each rule only within the scope it names, and follow stated ordering ("do X before Y").
- It is guidance, not a gate: if it doesn't cover your request, proceed with normal judgment and the browser actions.

## Article catalog

{{INDEX_MD}}
