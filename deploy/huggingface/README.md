---
title: NGS Explorer
emoji: 🏈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Explore Next Gen Stats tracking data in depth
---

# NGS Explorer

Every pass, route and carry chart Next Gen Stats publishes, read from the chart images live and matched to its
play-by-play row - built on the [next-gen-scrapy](https://github.com/rj7002/next-gen-scrapy) package.

Pick a player and season; the first load of a season downloads its play-by-play, so it takes a little longer.
Optional: set a `GEMINI_API_KEY` secret in the Space settings for the plain-English "ask" box.
