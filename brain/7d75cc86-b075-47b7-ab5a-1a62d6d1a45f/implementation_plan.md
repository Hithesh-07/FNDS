# Implementation Plan — Model Calibration & Balancing

Address the "underconfidence" issue where too many articles are being labeled as `UNCERTAIN`.

## Proposed Changes

### [Component] Prediction Logic (`predict.py` & `bert_predict.py`)

#### [MODIFY] [predict.py](file:///c:/Users/98858/.gemini/antigravity/playground/golden-apollo/predict.py)
- **Gap Reduction**: Change the UNCERTAIN gap threshold from `25%` to `15%`.
- **Threshold Adjustment**: Ensure the "LOW" confidence / UNCERTAIN floor is closer to `55%`.
- **Rule Relaxation**:
  # Rebrand to Aletheia & UI Color Overhaul

This task involves rebranding the "TruthLens" project to **Aletheia** and significantly improving the UI color palette, with a focus on fixing visibility issues in Dark Mode.

## User Review Required

> [!IMPORTANT]
> The rebranding will change the name displayed across the entire application. Please confirm if "Aletheia Pro" or just "Aletheia" is preferred (I will use "Aletheia Pro" for consistency with the previous version unless told otherwise).

## Proposed Changes

### [Branding]
#### [MODIFY] [index.html](file:///c:/Users/98858/.gemini/antigravity/playground/golden-apollo/templates/index.html)
- Replace all instances of `TruthLens` with `Aletheia`.
- Update the footer to reflect the new version.

### [UI/UX]
#### [MODIFY] [style.css](file:///c:/Users/98858/.gemini/antigravity/playground/golden-apollo/static/style.css)
- **Primary Color Shift**: Switch from generic blue/cyan to a premium "Electric Indigo" (`#6366f1`) and "Moonlight Cyan" (`#22d3ee`) palette.
- **Dark Mode Fix**:
    - Improve background darkness (`#0a0b14`) and card surface contrast (`#161b2c`).
    - Increase text brightness for primary content (`#ffffff`).
    - Enhance borders for better section definition.
- **Light Mode Refinement**:
    - Ensure a clean, professional "Enlightened" look (`#f8fafc` background).
- **Result Theming**: Calibrate Real/Fake/Uncertain colors for maximum pop in both modes.

## Verification Plan

### Automated Tests
- Run `final_test.py` to ensure core engine logic remains untouched.

### Manual Verification
- Use the `browser_subagent` to visualize both modes and ensure the "Aletheia" name is correctly displayed.
- Check contrast ratios for legal compliance and readability.
 load_model, predict; m,v,s=load_model(); print(predict('Cure for cancer suppressed by government hiding truth.', m,v,s)['label'])"` -> Expect `FAKE` (triggers >= 2)
2. **Full Suite**:
   - `python test_model.py`
   - `python bert_test.py`
- Test with the 3 specific cases provided by the user in the prompt to ensure the "Balance" is achieved.
