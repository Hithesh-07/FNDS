import os
import re

base_dir = r"C:\Users\98858\.gemini\antigravity"

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

# 1. Patch bert_predict.py
bert_path = os.path.join(base_dir, "bert_predict.py")
bert_content = read_file(bert_path)
# Update HF URL
bert_content = bert_content.replace(
    "api-inference.huggingface.co/models/",
    "router.huggingface.co/hf-inference/models/"
)
# Update retries & timeout
bert_content = bert_content.replace("for attempt in range(3):", "for attempt in range(2):")
bert_content = bert_content.replace("timeout = 25", "timeout = 12")

# Insert threading at the top and the timeout wrapper at the end
if "import threading" not in bert_content:
    bert_content = bert_content.replace("import time\n", "import time\nimport threading\n")
    
    timeout_wrapper = """

def bert_predict_with_timeout(text: str, timeout_seconds: int = 20) -> dict:
    result    = [None]
    exception = [None]

    def run():
        try:
            result[0] = bert_predict(text)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise Exception("BERT timeout after 20s — using SVM")
    if exception[0]:
        raise exception[0]
    return result[0]
"""
    bert_content += timeout_wrapper
write_file(bert_path, bert_content)


# 2. Patch app.py
app_path = os.path.join(base_dir, "app.py")
app_content = read_file(app_path)
if "from bert_predict import bert_predict_with_timeout" not in app_content:
    app_content = app_content.replace(
        "from bert_predict import bert_predict",
        "from bert_predict import bert_predict, bert_predict_with_timeout"
    )
    app_content = app_content.replace(
        "bert_result = bert_predict(text)",
        "bert_result = bert_predict_with_timeout(text, timeout_seconds=20)"
    )
write_file(app_path, app_content)


# 3. Patch static/style.css
css_path = os.path.join(base_dir, "static", "style.css")
css_content = read_file(css_path)
css_split = css_content.split("/* ── Reset ─────────────────────────────────────────────── */")
if len(css_split) > 1:
    css_content = css_split[0] # keep everything before the reset if any, usually reset is at the top
else:
    css_content = ""

user_css = r'''
/* ── Reset ─────────────────────────────────────────────── */
*, *::before, *::after {
    box-sizing : border-box;
    margin     : 0;
    padding    : 0;
}

body {
    background  : #080d14;
    color       : #e0e6f0;
    font-family : 'Inter', 'Segoe UI', sans-serif;
    min-height  : 100vh;
    overflow-x  : hidden;
}

/* ── Main container ─────────────────────────────────────── */
.container {
    max-width  : 900px;
    margin     : 0 auto;
    padding    : 2rem 1.5rem;
}

/* ── Result section ─────────────────────────────────────── */
.result-section {
    display        : flex;
    flex-direction : column;
    gap            : 1.2rem;
    margin-top     : 2rem;
    width          : 100%;
}

/* ── Verdict card ───────────────────────────────────────── */
.verdict-card {
    background    : #0d1421;
    border        : 1px solid #1a2535;
    border-radius : 16px;
    padding       : 2rem;
    display       : flex;
    align-items   : center;
    justify-content: space-between;
    flex-wrap     : wrap;
    gap           : 1rem;
    width         : 100%;
}

.verdict-label {
    font-size   : 3.5rem;
    font-weight : 900;
    letter-spacing: 0.05em;
    line-height : 1;
}
.verdict-real { color: #00ff88; text-shadow: 0 0 30px #00ff8840; }
.verdict-fake { color: #ff4757; text-shadow: 0 0 30px #ff475740; }
.verdict-uncertain { color: #ffa502; text-shadow: 0 0 30px #ffa50240; }

.verdict-confidence {
    text-align : right;
}
.confidence-number {
    font-size   : 2.5rem;
    font-weight : 800;
    color       : #00ff88;
}
.confidence-label {
    font-size      : 0.7rem;
    letter-spacing : 0.2em;
    color          : #ffffff40;
    display        : block;
    margin-top     : 0.2rem;
}

/* ── Two column grid ────────────────────────────────────── */
.two-col-grid {
    display               : grid;
    grid-template-columns : 1fr 1fr;
    gap                   : 1.2rem;
    width                 : 100%;
}

/* ── Info cards ─────────────────────────────────────────── */
.info-card {
    background    : #0d1421;
    border        : 1px solid #1a2535;
    border-radius : 12px;
    padding       : 1.5rem;
    width         : 100%;
}

.card-title {
    font-size      : 0.72rem;
    font-weight    : 700;
    letter-spacing : 0.2em;
    color          : #ffffff50;
    text-transform : uppercase;
    margin-bottom  : 1rem;
    display        : flex;
    align-items    : center;
    gap            : 0.5rem;
}

/* ── Probability bars ───────────────────────────────────── */
.prob-row {
    display     : flex;
    align-items : center;
    gap         : 0.8rem;
    margin-bottom: 0.7rem;
}
.prob-label {
    font-size      : 0.72rem;
    font-family    : monospace;
    letter-spacing : 0.1em;
    width          : 36px;
    color          : #ffffff60;
}
.prob-track {
    flex          : 1;
    height        : 6px;
    background    : #ffffff10;
    border-radius : 3px;
    overflow      : hidden;
}
.prob-fill {
    height        : 100%;
    border-radius : 3px;
    transition    : width 1s cubic-bezier(.22,1,.36,1);
}
.prob-fill-fake { background: linear-gradient(90deg, #ff4757, #ff6b7a); }
.prob-fill-real { background: linear-gradient(90deg, #00c896, #00ff88); }
.prob-pct {
    font-size   : 0.72rem;
    font-family : monospace;
    width       : 42px;
    text-align  : right;
    color       : #ffffff80;
}

/* ── Net score ──────────────────────────────────────────── */
.net-score {
    font-size   : 2rem;
    font-weight : 800;
    font-family : monospace;
}
.score-positive { color: #ff4757; }
.score-negative { color: #00ff88; }
.score-zero     { color: #ffffff40; }

/* ── Model table ────────────────────────────────────────── */
.model-table-card {
    background    : #0d1421;
    border        : 1px solid #1a2535;
    border-radius : 12px;
    padding       : 1.5rem;
    width         : 100%;
    overflow-x    : auto;
}

.model-table {
    width           : 100%;
    border-collapse : collapse;
    font-size       : 0.82rem;
}
.model-table th {
    text-align     : left;
    padding        : 0.6rem 1rem;
    border-bottom  : 1px solid #ffffff10;
    font-size      : 0.65rem;
    letter-spacing : 0.15em;
    color          : #ffffff30;
    font-weight    : 600;
}
.model-table td {
    padding        : 0.8rem 1rem;
    border-bottom  : 1px solid #ffffff05;
    vertical-align : middle;
}
.model-table tr:last-child td { border-bottom: none; }

.row-bert-active td { opacity: 1; }
.row-bert-failed td { opacity: 0.35; }
.row-svm td         { opacity: 0.75; }
.row-final td       {
    background : rgba(255,215,0,0.04);
    font-weight: 700;
}

.role-badge {
    font-size      : 0.62rem;
    font-family    : monospace;
    letter-spacing : 0.1em;
    padding        : 0.2rem 0.5rem;
    border-radius  : 4px;
    display        : inline-block;
}
.role-primary  { background: #00c8ff20; color: #00c8ff; border: 1px solid #00c8ff40; }
.role-fallback { background: #ffa50220; color: #ffa502; border: 1px solid #ffa50240; }
.role-failed   { background: #ff475720; color: #ff4757; border: 1px solid #ff475740; }
.role-engine   { background: #ffd70020; color: #ffd700; border: 1px solid #ffd70040; }

.text-fake      { color: #ff4757; font-weight: 700; }
.text-real      { color: #00ff88; font-weight: 700; }
.text-uncertain { color: #ffa502; font-weight: 700; }
.text-dim       { color: #ffffff25; }

/* ── Keywords ───────────────────────────────────────────── */
.keywords-grid {
    display   : flex;
    flex-wrap : wrap;
    gap       : 0.5rem;
    margin-top: 0.5rem;
}
.keyword-tag {
    font-size      : 0.75rem;
    font-family    : monospace;
    padding        : 0.25rem 0.6rem;
    border-radius  : 4px;
    background     : #ffffff08;
    border         : 1px solid #ffffff15;
    color          : #ffffff70;
}
.keyword-fake { border-color: #ff475740; color: #ff6b7a; background: #ff475710; }
.keyword-real { border-color: #00ff8840; color: #00ff88; background: #00ff8810; }

/* ── Disclaimer ─────────────────────────────────────────── */
.disclaimer {
    text-align  : center;
    font-size   : 0.75rem;
    color       : #ffffff25;
    padding     : 1rem 0;
    font-style  : italic;
}

/* ── Responsive ─────────────────────────────────────────── */
@media (max-width: 640px) {
    .two-col-grid {
        grid-template-columns : 1fr;
    }
    .verdict-label {
        font-size : 2.5rem;
    }
    .confidence-number {
        font-size : 1.8rem;
    }
    .model-table {
        font-size : 0.72rem;
    }
    .model-table th,
    .model-table td {
        padding : 0.5rem 0.5rem;
    }
}
'''

# The original CSS had loading animation styles and highlights before reset? 
# I will append loading animation back to be safe. But `style.css` looks completely replaced.
# Let's keep existing css from original and just append using the user's snippet where it clashes!
# Wait, user explicitly says "Replace entire style.css layout section with this fixed version".
# So `css_content += user_css` is perfect.
write_file(css_path, css_content + user_css)


# 4. Patch index.html
html_path = os.path.join(base_dir, "templates", "index.html")
html_content = read_file(html_path)

user_html = r"""<section id="results-section" class="results-section hidden">
<div class="result-section">

    <!-- Verdict Card -->
    <div class="verdict-card">
        <div>
            <div class="verdict-label
                {% if result and result.label == 'FAKE' %}verdict-fake
                {% elif result and result.label == 'REAL' %}verdict-real
                {% else %}verdict-uncertain{% endif %}" id="verdict-label">
                {{ result.label if result else '---' }}
            </div>
            <div style="font-size:0.72rem; letter-spacing:0.15em;
                        color:#ffffff40; margin-top:0.5rem;">
                AI CONFIDENCE LEVEL
            </div>
        </div>
        <div class="verdict-confidence">
            <div class="confidence-number" id="confidence-badge-val">{{ result.confidence if result else '0' }}%</div>
            <span class="confidence-label" id="verdict-tier">{{ result.confidence_level if result else 'CERTAINTY' }}</span>
            <div style="margin-top:0.8rem;">
                <span id="role-badge-primary" class="role-badge
                    {% if result and result.bert_ok %}role-primary{% else %}role-fallback{% endif %}">
                    {% if result and result.bert_ok %}🤖 BERT Active{% else %}⚙️ SVM Active{% endif %}
                </span>
            </div>
        </div>
    </div>

    <!-- Two Column Grid -->
    <div class="two-col-grid">

        <!-- Priority Engine -->
        <div class="info-card">
            <div class="card-title">⚡ Priority Engine Logic</div>
            <div style="font-size:0.8rem; color:#00c8ff;
                        font-family:monospace; margin-bottom:1rem;" id="logic-reason">
                {{ result.decision_reason if result else 'ML_CONSENSUS' }}
            </div>
            
            <div class="prob-row">
                <span class="prob-label">FAKE</span>
                <div class="prob-track">
                    <div class="prob-fill prob-fill-fake" id="bar-fake"
                         style="width:{{ result.fake_prob if result else '0' }}%"></div>
                </div>
                <span class="prob-pct" id="p-fake">{{ result.fake_prob if result else '0' }}%</span>
            </div>
            
            <!-- We also need the logic-bar-fill for JS fallback if it relies on logic-bar-fill mapping -->
            <!-- Wait, user JS accesses id="logic-bar-fill", I will add it hidden here just so it doesn't break JS -->
            <div id="logic-bar-fill" style="display:none;"></div>
            
            <div class="prob-row">
                <span class="prob-label">REAL</span>
                <div class="prob-track">
                    <div class="prob-fill prob-fill-real" id="bar-real"
                         style="width:{{ result.real_prob if result else '0' }}%"></div>
                </div>
                <span class="prob-pct" id="p-real">{{ result.real_prob if result else '0' }}%</span>
            </div>
        </div>

        <!-- Credibility Signals -->
        <div class="info-card">
            <div class="card-title">🔍 Credibility Signals</div>
            <div style="margin-bottom:0.5rem; font-size:0.72rem;
                        color:#ffffff40; letter-spacing:0.1em;">
                NET SCORE
            </div>
            <div class="net-score
                {% if result and result.net_score > 0 %}score-positive
                {% elif result and result.net_score < 0 %}score-negative
                {% else %}score-zero{% endif %}" id="net-score-val">
                {% if result and result.net_score > 0 %}+{% endif %}{{ result.net_score if result else '0' }}
            </div>
            <div style="margin-top:1rem;" id="cred-flags-grid">
            {% if result and result.credibility_flags %}
                {% for flag in result.credibility_flags[:4] %}
                <div style="font-size:0.7rem; font-family:monospace;
                            color:#ff6b7a; margin-bottom:0.3rem;">
                    ⚠ {{ flag }}
                </div>
                {% endfor %}
            {% endif %}
            </div>
        </div>

    </div>

    <!-- Model Table -->
    <div class="model-table-card" id="model-consensus-card" style="display:block;">
        <div class="card-title">⚖️ Model Details</div>
        <table class="model-table">
            <thead>
                <tr>
                    <th>MODEL</th>
                    <th>ROLE</th>
                    <th>VERDICT</th>
                    <th>FAKE %</th>
                    <th>REAL %</th>
                    <th>CONFIDENCE</th>
                </tr>
            </thead>
            <tbody id="model-table-body">
                <!-- Fallback SSR content in case no JS -->
            </tbody>
        </table>
    </div>

    <!-- Keywords -->
    <div class="info-card" id="keywords-container" style="display:none;">
        <div class="card-title">🔑 Key Signals</div>
        <div class="keywords-grid" id="keywords-grid">
            {% if result and result.keywords %}
            {% for kw in result.keywords %}
            <span class="keyword-tag">{{ kw }}</span>
            {% endfor %}
            {% endif %}
        </div>
    </div>

    <!-- Keyword Highlights Legend -->
    <div id="highlight-legend" class="highlight-legend hidden" style="margin-top:1rem;">
          <span class="legend-item"><span class="legend-dot legend-fake" style="background:var(--fake);width:8px;height:8px;border-radius:50%;display:inline-block;"></span> Fake signal words</span>
          <span class="legend-item"><span class="legend-dot legend-real" style="background:var(--real);width:8px;height:8px;border-radius:50%;display:inline-block;"></span> Credibility words</span>
          <small style="opacity:0.6;margin-left:10px;">(Click highlighted text above to dismiss)</small>
    </div>

    <!-- Disclaimer -->
    <div class="disclaimer">
        Always verify important news from trusted sources like
        BBC, Reuters, or AP News. AI analysis is not a substitute
        for human judgment. · <a href="/about"
        style="color:#ffffff40;">About Aletheia</a>
    </div>

</div>
</section>"""

# Replace the HTML <section id="results-section">...</section> 
pattern = re.compile(r'<section id="results-section".*?</section>', re.DOTALL)
html_content = re.sub(pattern, user_html, html_content)
write_file(html_path, html_content)

print("Patch applied.")
