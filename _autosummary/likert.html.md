# likert

Tools to analyze Likert scale data.

Likert-scale surveys produce ordered categorical responses (“strongly disagree”
through “strongly agree”) that need dedicated summarizing, scoring and plotting
tools. This package collects such tools, along with worked analyses of real
survey data.

The core package has no third-party dependencies. The worked analyses live in
the `likert.examples` subpackage and need the extra dependencies installed by
`pip install likert[examples]`.

See `likert.examples.in_their_own_words` for a complete study: cleaning
free-text survey answers, mapping them onto a controlled vocabulary, scoring
sentiment, and rendering word clouds and proportion plots.
