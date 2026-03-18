# Implementation Plan: TruthLens UI Upgrades

This document outlines the plan to implement the 9 requested UI upgrades for the TruthLens application.

## Proposed Changes

### UI Components
  
#### [MODIFY] [index.html](file:///c:/Users/98858/.gemini/antigravity/playground/golden-apollo/templates/index.html)
- **UPGRADE 6 (Dark Mode Toggle):** Add a 🌙/☀️ toggle button in top right corner. Update JavaScript to handle toggling `body.dark` and saving to `localStorage`.
- **UPGRADE 1 (Dynamic Backgrounds):** Update JS fetch callback to apply `body-fake`, `body-real`, or `body-uncertain` to `document.body`.
- **UPGRADE 2 (Loading Animation & Progress):** Add a Sweeping progress bar element. Modify the submit event listener to inject the `<span class="spinner"></span> Analyzing...` HTML, disable the button, and trigger the sweeping progress bar.
- **UPGRADE 4 (Credibility Gauge):** Replace the standard confidence percentage box with an SVG-based speedometer gauge. Update JS to calculate the rotation angle based on `data.confidence`, coloring the zones (red/yellow/green).
- **UPGRADE 3 (Probability Bars):** Update JS to set `--target-width` inline style instead of animating `width` via JS setTimeout, letting CSS handle the animation natively.
- **UPGRADE 5 (Recent Analysis History):** Add a new "Recent Checks" container below the main card. Add JS to read/write from `localStorage` (`tl_history`) and dynamically render the recent analyses list after each submission.
- **UPGRADE 7 (Download Result):** Include `html2canvas` via CDN. Add a "Download Result" button and the `downloadCard()` JS function.
- **UPGRADE 9 (Word Highlighting):** Add a new section below results to display the originally submitted text. Add JS logic to iterate over `data.keywords.fake` and `data.keywords.real` (if available), wrapping matched words in styled `<span>` tags.

### Styles
  
#### [MODIFY] [style.css](file:///c:/Users/98858/.gemini/antigravity/playground/golden-apollo/static/style.css)
- **UPGRADE 6 (CSS Variables & Dark Mode):** Introduce `--bg`, `--card-bg`, `--text`, `--subtext`, `--border` in `:root`. Add `body.dark` overrides. Find and replace hardcoded colors to use these variables where applicable in the main structure.
- **UPGRADE 1 (Dynamic Backgrounds):** Add `body.body-fake`, `body.body-real`, `body.body-uncertain` background gradient rules.
- **UPGRADE 2 & 3 (Animations):**
  - Add `.spinner` and `@keyframes spin`.
  - Add sweeping progress bar CSS.
  - Add `.verdict-stamp` and `@keyframes stampIn`.
  - Add `.bar-fill` and `@keyframes growBar` for probability bars.
- **UPGRADE 4 (Gauge Styles):** Add CSS for the SVG semicircle, needle, and center text positioning.
- **UPGRADE 5 (History Section):** Add styling for the recent checks list, items, dots, and timestamps.
- **UPGRADE 7 (Download Button):** Add CSS base styles for the new download button.
- **UPGRADE 9 (Word Highlighting):** Add highlighting classes (`.fake-word` in red and `.real-word` in green).
- **UPGRADE 8 (Responsive):** Implement `@media (max-width: 768px)` and `@media (max-width: 480px)` styles.

## Verification Plan

### Manual Verification
1. Start the Flask server locally.
2. Open `http://localhost:5000` in the browser.
3. Observe the loading animation (spinner + text + progress bar sweep) on submit.
4. Test a Fake News example to verify the dynamic background (red), verdict stamp animation, and probability bar animations.
5. Verify the SVG speedometer needle updates correctly based on the confidence percentage and points to the correct zone.
6. Check if the "Recent Analysis" localStorage history populated correctly below the card.
7. Toggle the Dark Mode switch and ensure colors update correctly.
8. Click the "Download Result" button and verify a clean PNG card is downloaded.
9. Verify that fake-/real-leaning keywords are highlighted in the original text shown appropriately below the result card.
10. Validate mobile layout in the browser's responsive design mode using DevTools.
