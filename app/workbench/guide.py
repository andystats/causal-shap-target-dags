"""Guide tab: the study guide, simplified and embedded.

The math renders via MathJax when online and degrades to readable TeX.
"""

from shiny import ui

AMBER = "#b45309"

_G = r"""
<style>
#guide-content table td { white-space: normal; max-width: none;
    font-family: Georgia, serif; font-size: 0.85rem; line-height: 1.5; }
#guide-content table th { white-space: normal; }
</style>
<div id="guide-content" style="max-width: 880px; margin: 0 auto; font-family: Georgia, serif;">

<div class="card">
<div class="card-title">The two questions</div>
<p>Ordinary SHAP answers an <em>associational</em> question:
<strong>&ldquo;what did knowing x<sub>i</sub> add to the prediction?&rdquo;</strong>
Causal SHAP answers an <em>interventional</em> one:
<strong>&ldquo;what would setting x<sub>i</sub> do to the outcome?&rdquo;</strong>
This Workbench is the machinery for moving responsibly from the first
question to the second &mdash; and for measuring what breaks when the graph
used for that move is wrong.</p>
</div>

<div class="card">
<div class="card-title">The shared Shapley skeleton</div>
<p>Both attributions average a feature's marginal contribution over
orderings:</p>
<p style="text-align:center;">\( \phi_i \;=\; \mathbb{E}_{\pi}\!\left[\,
v(P_i^{\pi} \cup \{i\}) - v(P_i^{\pi})\,\right] \)</p>
<p>where \(P_i^{\pi}\) is the set of features preceding \(i\) in ordering
\(\pi\). Everything that distinguishes the methods lives in two choices:
<strong>the value function</strong> \(v(S)\) and <strong>which orderings are
admissible</strong>.</p>
<table style="margin-top:8px;">
<tr><th></th><th>value function</th><th>orderings</th></tr>
<tr><td><strong>Ordinary SHAP</strong></td>
<td>\( v(S) = \mathbb{E}[\,f(X)\mid X_S = x_S\,] \) &mdash; conditioning</td>
<td>all \(n!\)</td></tr>
<tr><td><strong>Causal SHAP</strong></td>
<td>\( v(S) = \mathbb{E}[\,f(X)\mid \mathrm{do}(X_S = x_S)\,] \) &mdash;
graph surgery (Heskes 2020)</td>
<td>topological orders of \(G\) only (Frye 2020)</td></tr>
</table>
<p style="margin-top:10px;"><em>Shorthand: SHAP = all orderings &times;
conditioning. Causal SHAP = graph-consistent orderings &times; do().</em>
The do() expectation is estimated by forward simulation: fix the coalition,
sample every other node from \(\widehat{P}(X_j \mid \mathrm{pa}(X_j))\)
in topological order, average the model's predictions.</p>
<p><strong>What changes in practice:</strong> mediators stop absorbing their
parents' credit, proxies and colliders fall toward zero, root causes recover
credit transmitted through descendants. The toy trap: a clinic-visit proxy
took 45.6% of SHAP credit with a true causal share of 0%.</p>
</div>

<div class="card">
<div class="card-title">Concept glossary</div>
<table>
<tr><td><strong>CPDAG</strong></td><td>what discovery can actually identify:
an equivalence class with some edges left undirected</td></tr>
<tr><td><strong>do(&middot;)</strong></td><td>set a variable and sever its
incoming edges; distinct from conditioning</td></tr>
<tr><td><strong>Backdoor path</strong></td><td>a non-causal D&ndash;Y path
starting with an arrow into the exposure; the source of confounding</td></tr>
<tr><td><strong>Adjustment set Z</strong></td><td>blocks every backdoor path,
opens no collider, contains no descendant of D</td></tr>
<tr><td><strong>Shrier&ndash;Platt test</strong></td><td>the six-step
graphical check that Z blocks all backdoor paths</td></tr>
<tr><td><strong>Collider</strong></td><td>A &rarr; C &larr; B; conditioning
on C <em>opens</em> the path</td></tr>
<tr><td><strong>SHD</strong></td><td>structural Hamming distance: edge edits
separating two graphs</td></tr>
<tr><td><strong>Constraint ledger</strong></td><td>the versioned
forbidden/required-edge record of expert judgment</td></tr>
</table>
</div>

<div class="card">
<div class="card-title">The metric battery (Evaluate, station 4)</div>
<p>Two axes. <strong>Concordance</strong> asks whether the learned picture
matches the truth; <strong style="color:AMBER_C;">structural
importance</strong> asks whether the picture <em>works</em> for the target
relationship D &rarr; Y. They can dissociate &mdash; topological error and
functional failure are different events &mdash; and measuring that
dissociation is the program's novel contribution.</p>
<table>
<tr><td><strong>M1</strong></td><td>edge concordance &mdash; precision /
recall / F1 (directed + skeleton), SHD</td></tr>
<tr><td><strong>M2</strong></td><td>target-pathway scorecard &mdash; correct
/ reversed / missed / spurious on D&rarr;&hellip;&rarr;Y paths</td></tr>
<tr><td style="color:AMBER_C;"><strong>M3</strong></td>
<td style="color:AMBER_C;">sufficiency transfer &mdash; derive minimal
Z&prime; from the learned graph, test it in the sealed truth (backdoor
blocked <em>and</em> no descendant of D); Jaccard and excess size vs the
true minimal set</td></tr>
<tr><td style="color:AMBER_C;"><strong>M4</strong></td>
<td style="color:AMBER_C;">parameter fidelity &mdash; the D&rarr;Y effect
estimated under Z&prime;, bias vs the frozen truth: structure error priced
in outcome units</td></tr>
<tr><td style="color:AMBER_C;"><strong>M5</strong></td>
<td style="color:AMBER_C;">identification honesty &mdash; the fraction of
unique consistent CPDAG extensions under which Z&prime; remains a full valid
adjustment set; large orientation spaces use a labelled Monte Carlo sample</td></tr>
</table>
</div>

<div class="card">
<div class="card-title">Study flow</div>
<p><strong>0</strong> seal the known world (locked graph, generator, frozen
do()-truth) &middot; <strong>I</strong> discovery on structure-blind data
&rarr; CPDAG ensemble &middot; <strong>II</strong> expert rounds via the
constraint ledger &rarr; plausible DAG G&prime; &middot;
<strong>III</strong> dual evaluation (M1&ndash;M5, per algorithm &times;
round) &middot; <strong>IV</strong> attribution under G vs G&prime; &middot;
<strong>V</strong> complexity companion &middot; <strong>VI</strong>
robustness.</p>
<p style="font-size:0.9em;color:#555;">References: Heskes et al. 2020
(NeurIPS); Frye et al. 2020;
Shrier &amp; Platt 2008; VanderWeele &amp; Shpitser 2011.</p>
</div>

</div>
""".replace("AMBER_C", AMBER)


def guide_panel():
    return ui.nav_panel("Guide", ui.HTML(_G))
