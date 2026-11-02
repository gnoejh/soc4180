-- Write TeX into the deck as \( \) / \[ \] text for the embedded MathJax 3
-- (mathjax.html) to typeset, instead of Quarto's reveal.js math plugin,
-- which downloads MathJax 2 from a CDN when the deck opens: offline, or in a
-- preview that blocks outside scripts, every equation then shows as raw TeX.
-- Use with `html-math-method: plain` so Quarto adds no loader of its own.
-- Only reveal.js output is touched; lab.ipynb keeps ordinary $ math.

local function escape(s)
  return (s:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;"))
end

function Math(el)
  if not FORMAT:match("revealjs") then
    return nil
  end
  if el.mathtype == "DisplayMath" then
    return pandoc.RawInline("html", '<span class="math display">\\[' .. escape(el.text) .. "\\]</span>")
  end
  return pandoc.RawInline("html", '<span class="math inline">\\(' .. escape(el.text) .. "\\)</span>")
end
