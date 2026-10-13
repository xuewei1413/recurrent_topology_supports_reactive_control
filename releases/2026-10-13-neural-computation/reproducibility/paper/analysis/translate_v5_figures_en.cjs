/* Translate editable SVG labels without rebuilding scientific geometry. */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { pathToFileURL } = require('url');
const { chromium } = require('C:/Users/30491/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const ROOT = path.resolve(__dirname, '../..');
const SOURCE = path.join(ROOT, 'paper/figures/v5/chinese_final');
const OUTPUT = path.join(ROOT, 'paper/figures/v5/english_final');
const BROWSER = 'C:/Users/30491/AppData/Local/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-win64/chrome-headless-shell.exe';
const FILES = ['00_overview', '01_cross_perturbation', '02_topology_controls', '03_localization', '04_conditional_retention', '05_readout_boundaries'];
const normalize = s => s.replace(/\s/g, '');
const LABELS = {
  '本地复跑主队列': 'Local rerun: main cohort',
  '本地复跑后续队列': 'Local rerun: subsequent cohort',
  '历史原计划 15 个 + 追加 15 个种子': '15 historically planned + 15 added seeds',
  '● 原计划范围': '● Planned range',
  '◇ 追加范围': '◇ Added range',
  '结构干预': 'Connectivity interventions',
  '输出端的积分与动作读出': 'Output integration and action readout',
  '固定突触幅值，仅进化连接': 'Fixed weight magnitudes; evolved connectivity',
  '任务观测': 'Observations',
  '原网络': 'Original network',
  '背景重接': 'Background rewiring',
  '保持边数与入度、出度': 'Edge count, in- and out-degree preserved',
  '指定连接删除': 'Specified deletion',
  '按次序累计删除连接组': 'Sequential connection-group deletion',
  '计算过程示意 · 非实验轨迹': 'Schematic computation; not recorded trajectories',
  '汇聚输入': 'Input currents',
  '膜电位': 'Membrane potential',
  '发放与计数': 'Spikes and counts',
  '阈值': 'Threshold',
  '内部时间': 'Internal time',
  '+ 兴奋': '+ Excitation',
  '− 抑制': '- Inhibition',
  '动作': 'Action',
  '比较计数': 'Compare counts',
  '20 个内部步 · 包含平局裁决': '20 internal steps; tie-breaking included',
  '重接与删除后的功能保持相关': 'Association between performance after rewiring and deletion',
  'CartPole · 25 个原网络': 'CartPole: 25 original networks',
  '每点为同一原网络的配对测量': 'Each point: paired measures of one network',
  '实测关联': 'Measured association',
  '待检验的解释联系': 'Candidate explanation',
  '结构改变后的输出活动': 'Output activity after connectivity changes',
  '可能解释网络的耐受差异': 'May explain differences in tolerance',
  '机制解释': 'Mechanistic interpretation',
  '背景重接后保留的回报比例': 'Return retained after background rewiring',
  '指定删除后的回报': 'Return after specified deletion',
  '重接后功能保持B': 'Normalized post-rewiring return',
  '删除后回报J24': 'Post-deletion return J24',
  '相关系数ρ': 'Correlation coefficient rho',
  '未调整': 'Unadjusted',
  '控剂量': 'Deletion-adjusted',
  '完整调整': 'Fully adjusted',
  '主队列': 'Main cohort',
  '另一队列': 'Subsequent cohort',
  '新拓扑主队列': 'Newly evolved main cohort',
  '25 个网络': '25 networks',
  '另一拓扑队列': 'Subsequent cohort',
  '15 个原定网络 + 15 个结果后新增网络': '15 planned + 15 added after results',
  '● 原定': '● Planned',
  '◇ 结果后新增': '◇ Added after results',
  '每点为一个网络': 'Each point: one network',
  '新评估段复测': 'New-episode retest',
  '沿用 b 的 30 个网络': 'Same 30 networks as b',
  '跨评估段配对': 'Cross-episode pairing',
  '原评估段 B / 新评估段回报': 'Earlier rewiring / new deletion',
  '协变量调整': 'Covariate adjustment',
  '点估计与 95% CI': 'Estimates and 95% CI',
  '复测不增加独立网络数；完整调整包含基准回报、输出敏感性与实际删边比例。': 'Retesting adds no networks. Full adjustment: baseline return, output sensitivity, deletion fraction.',
  '具体连接组织支撑控制表现': 'Connection arrangement supports control',
  '同一进化网络与两类随机化结构配对比较 · 每任务 20 个拓扑': 'Paired comparisons with two randomized controls; 20 networks per task',
  '同边数随机图': 'Same edge count',
  '度保持重接图': 'Degree preserved',
  '原网络中位回报 499.4': 'Median original return 499.4',
  '原网络中位回报 -90.3': 'Median original return -90.3',
  '原网络中位回报 -112.7': 'Median original return -112.7',
  '原网络中位回报 -128.7': 'Median original return -128.7',
  '两项优势均超过预设差': 'Both exceed threshold',
  '横轴：配对回报差的中位数；每行单独标尺。线段为 95% bootstrap 区间。': 'Median paired return differences; separate row scales. Lines: 95% bootstrap intervals.',
  '相同删边数，行为损伤仍因选边方式而异': 'Equal deletion counts can yield different impairment',
  'CartPole · 每点为一条进化谱系；纵向散开仅用于避免点重叠': 'CartPole: each point is one lineage; vertical jitter reveals overlap',
  '按三跳路径分数选边': 'Three-hop path score',
  '高分边 vs 匹配边；30 条谱系': 'High-score vs matched; 30 lineages',
  '预设回报差 30': 'Prespecified difference 30',
  '中位超额损伤  158.6': 'Median excess impairment 158.6',
  '损伤差为正  25/30': 'Positive differences: 25/30',
  '输出响应计分': 'Output sensitivity',
  '高敏感边 vs 低敏感边；30 条谱系': 'High vs low sensitivity; 30 lineages',
  '中位超额损伤  44.1': 'Median excess impairment 44.1',
  '横轴为等边数删除的超额回报损伤；零线表示超额损伤为零。两面板标尺不同。': 'Excess return loss at equal deletion counts; zero indicates no excess. Panels use separate scales.',
  '匹配计算路径分数的两项连接数量后，可比较的边减少；精确匹配结果见正文。': 'Matching both path-score component counts reduces eligible edges; exact-matching results are in the text.',
  '仅保留易损连接，通常低于完整网络表现': 'Retaining vulnerable connections alone usually lowers performance',
  'CartPole 全因子队列 · 30 条谱系，各谱系先合并两个预算版本': 'Full-factorial CartPole cohort: 30 lineages; two budget versions averaged per lineage',
  'A  四种保留条件下的回报': 'A  Return under four retention conditions',
  '完整网络': 'Complete network',
  '中位 499.5': 'Median 499.5',
  '只保留高分边集': 'High-score connections only',
  '中位 276.4': 'Median 276.4',
  '只保留其余隐藏边': 'Other hidden connections only',
  '中位 229.3': 'Median 229.3',
  '仅保留输入输出连接': 'Input-output connections only',
  '中位 270.4': 'Median 270.4',
  'B  删除交互': 'B  Deletion interaction',
  '中位 266.7': 'Median 266.7',
  '两次单独删除的损失之和 > 联合删除损失': 'Sum of separate losses > joint loss',
  '仅输入输出连接优于部分保留': 'Interface-only > at least one partial set',
  'CartPole · 指定边集的条件比较': 'CartPole: comparisons for the specified sets',
  '同一连接图会受读出规则影响': 'Readout rules affect the same connectivity graph',
  '改规则损伤已有策略；按新规则重新进化，结构优势仍可出现': 'Changing the rule impairs existing policies; evolution under the new rule retains an advantage',
  'A  原网络交叉测试：改用随机平局': 'A  Existing policies tested with random tie-breaking',
  '网络回报下降': 'Networks with lower return',
  '中位损失 459.7': 'Median loss 459.7',
  '横轴：默认最小动作编号读出回报 − 随机平局读出回报；每点为一个原网络。': 'Smallest-index return minus random-tie return; each point is one original network.',
  'B  在随机平局规则下重新进化': 'B  New evolution under random tie-breaking',
  '横轴：相对度保持重接对照的配对回报差；两任务分别标尺。': 'Paired return advantages over degree-preserving controls; separate task scales.',
  '超过两类对照': 'Exceeds both controls',
  'C  证据覆盖范围': 'C  Tasks supporting each finding',
  '随机结构对照': 'Randomized controls',
  '支持': 'Supported',
  '15/20 · 有限': '15/20; limited',
  '按路径分数选边': 'Path-score selection',
  '较大效应': 'Larger effect',
  '未报告': 'Not reported',
  '按输出变化选边': 'Output-sensitivity selection',
  '超过预设差': 'Above threshold',
  '低于预设差': 'Below threshold',
  '跨扰动关联': 'Cross-perturbation association',
  '多队列同向': 'Across cohorts',
  '任务内初步': 'Preliminary',
  '随机平局重进化': 'Random-tie evolution',
  '“未报告”表示本稿未呈现对应检验；不同任务的状态不构成任务间效应比较。': 'Not reported: no corresponding test presented. Task-specific statuses do not compare effects across tasks.'
};
const MAPPING = Object.fromEntries(Object.entries(LABELS).map(([a, b]) => [normalize(a), b]));
const hash = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');

async function main() {
  fs.mkdirSync(OUTPUT, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: BROWSER });
  const records = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1800, height: 1600 }, deviceScaleFactor: 1 });
    for (const stem of FILES) {
      const source = path.join(SOURCE, stem + '.svg');
      await page.goto(pathToFileURL(source).href);
      await page.evaluate(() => document.fonts.ready);
      const record = await page.evaluate(({ mapping, stem }) => {
        const svg = document.querySelector('svg');
        const signature = () => Array.from(svg.querySelectorAll('path,line,rect,circle,ellipse,polygon,polyline,use,image'))
          .map(e => [e.tagName, Array.from(e.attributes).map(a => [a.name, a.value])]);
        const before = JSON.stringify(signature());
        const translated = [];
        for (const el of svg.querySelectorAll('text')) {
          const original = el.textContent.replace(/\s/g, '');
          if (!/[\u4e00-\u9fff]/.test(original)) continue;
          const label = mapping[original];
          if (!label) throw new Error('Missing translation: ' + original);
          const bbox = el.getBBox();
          const leaf = el.querySelector('tspan') || el;
          const css = getComputedStyle(leaf);
          const fontSize = parseFloat(css.fontSize);
          const fill = css.fill, weight = css.fontWeight, fontStyle = css.fontStyle;
          if (el.children.length) {
            const firstY = leaf.getAttribute('y');
            el.setAttribute('x', String(bbox.x + bbox.width / 2));
            el.setAttribute('y', firstY || String(bbox.y + bbox.height * .8));
            el.style.textAnchor = 'middle';
          }
          el.textContent = label;
          el.style.fontFamily = 'Arial, Helvetica, sans-serif';
          el.style.fontSize = fontSize + 'px';
          el.style.fontWeight = weight;
          el.style.fontStyle = fontStyle;
          el.style.fill = fill;
          if (stem === '01_cross_perturbation' && (original === '主队列' || original === '另一队列')) {
            el.setAttribute('x', String(parseFloat(el.getAttribute('x')) + 25));
            el.style.paintOrder = 'stroke fill';
            el.style.stroke = 'white';
            el.style.strokeWidth = '1.8px';
          }
          let maxWidth;
          const x = parseFloat(el.getAttribute('x')), y = parseFloat(el.getAttribute('y'));
          if (stem === '00_overview') {
            maxWidth = y < 30 ? (x < 300 ? 245 : 340) : 230;
            if (x < 300 && y > 230 && y < 280) maxWidth = 136;
            if (x >= 510) maxWidth = 166;
            if (x >= 297 && x < 510 && y > 300) maxWidth = 193;
            if (y > 280 && y < 315 && x < 300) maxWidth = 454;
            if (original === '固定突触幅值，仅进化连接') maxWidth = 270;
            if (original === '计算过程示意·非实验轨迹') maxWidth = 350;
          } else if (stem === '01_cross_perturbation') {
            maxWidth = y > 400 ? 480 : 215;
            if (y >= 245 && y < 280) maxWidth = 147;
            if (original === '控剂量') maxWidth = 75;
            if (original === '未调整' || original === '完整调整') maxWidth = 75;
            if (original.includes('J24')) maxWidth = 160;
          } else {
            maxWidth = 1600;
            if (stem === '02_topology_controls' && y > 250 && y < 950) maxWidth = x < 500 ? 410 : 320;
            if (stem === '03_localization' && y > 200 && y < 800) maxWidth = 690;
            if (stem === '04_conditional_retention' && y > 735 && y < 815) maxWidth = 405;
            if (stem === '04_conditional_retention' && original.startsWith('B删除交互')) maxWidth = 270;
            if (stem === '04_conditional_retention' && original.startsWith('仅输入输出连接优于')) maxWidth = 515;
            if (stem === '05_readout_boundaries' && x > 1300 && y < 1000) maxWidth = 340;
            if (stem === '05_readout_boundaries' && y > 1120 && y < 1400) maxWidth = x < 500 ? 450 : 238;
          }
          const width = el.getBBox().width;
          if (width > maxWidth) el.style.fontSize = (fontSize * maxWidth / width) + 'px';
          translated.push({ original, english: label, font_size: parseFloat(el.style.fontSize), max_width: maxWidth });
        }
        for (const el of svg.querySelectorAll('text')) {
          if (/[\u4e00-\u9fff]/.test(el.textContent)) throw new Error('Chinese label remains');
        }
        for (const el of svg.querySelectorAll('title,desc')) {
          if (/[\u4e00-\u9fff]/.test(el.textContent)) el.textContent = 'English manuscript figure: ' + stem;
        }
        if (before !== JSON.stringify(signature())) throw new Error('Scientific geometry changed');
        return { translated, scientific_geometry_unchanged: true, svg: svg.outerHTML,
                 width: svg.viewBox.baseVal.width, height: svg.viewBox.baseVal.height };
      }, { mapping: MAPPING, stem });
      await page.evaluate(() => document.fonts.ready);
      fs.writeFileSync(path.join(OUTPUT, stem + '.svg'), record.svg + '\n', 'utf8');
      const height = Math.round(1800 * record.height / record.width);
      await page.setViewportSize({ width: 1800, height });
      await page.evaluate(({ height }) => {
        const svg = document.querySelector('svg');
        svg.setAttribute('width', '1800');
        svg.setAttribute('height', String(height));
      }, { height });
      await page.locator('svg').screenshot({ path: path.join(OUTPUT, stem + '.png') });
      const { svg, ...details } = record;
      records.push({ file: stem, source_sha256: hash(source), output_svg_sha256: hash(path.join(OUTPUT, stem + '.svg')),
                     output_png_sha256: hash(path.join(OUTPUT, stem + '.png')), ...details });
      process.stdout.write(stem + ': ' + record.translated.length + ' labels translated\n');
    }
  } finally {
    await browser.close();
  }
  fs.writeFileSync(path.join(OUTPUT, 'manifest.json'), JSON.stringify({
    operation: 'English label translation; scientific SVG geometry and plotted values preserved',
    source_manuscript_sha256: hash(path.join(ROOT, 'paper/manuscript_v5_zh_draft.md')),
    english_manuscript_sha256: hash(path.join(ROOT, 'paper/manuscript_v5_en_draft.md')),
    axis_label_note: 'The old undefined B shorthand is spelled out as normalized post-rewiring return; J24 is retained.',
    figures: records
  }, null, 2) + '\n');
}
main().catch(e => { process.stderr.write(e.stack + '\n'); process.exit(1); });
