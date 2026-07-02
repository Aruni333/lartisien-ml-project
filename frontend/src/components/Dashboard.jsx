import leaderboard from '../data/leaderboard.json'

const GOLD   = '#C9A02C'
const GOLD_L = '#E8C96A'

const s = {
  sectionTitle: {
    fontSize: 11, color: GOLD, letterSpacing: '0.22em', textTransform: 'uppercase',
    marginBottom: 24, paddingBottom: 10, borderBottom: `1px solid rgba(201,160,44,0.2)`,
    fontWeight: 600,
  },
  statsGrid: { display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16, marginBottom: 36 },
  statCard: {
    background: '#0D0D0D', border: `1px solid rgba(201,160,44,0.22)`,
    borderRadius: 10, padding: '20px 16px', textAlign: 'center',
  },
  statLabel: { fontSize: 9, letterSpacing: '0.2em', textTransform: 'uppercase', color: '#555', marginBottom: 8 },
  statValue: { fontSize: 24, fontWeight: '700', color: GOLD },
  statSub:   { fontSize: 10, color: '#444', marginTop: 5, letterSpacing: '0.05em' },
  grid2: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 28 },
  card: {
    background: '#0D0D0D', border: `1px solid rgba(201,160,44,0.18)`,
    borderRadius: 10, padding: 24,
  },
  cardTitle: { fontSize: 10, fontWeight: '600', color: GOLD, marginBottom: 18, letterSpacing: '0.18em', textTransform: 'uppercase' },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 12 },
  th: {
    textAlign: 'left', padding: '8px 10px', fontSize: 9, letterSpacing: '0.18em',
    textTransform: 'uppercase', color: '#555',
    borderBottom: `1px solid rgba(201,160,44,0.18)`,
  },
  td: (winner) => ({
    padding: '10px 10px', borderBottom: '1px solid rgba(255,255,255,0.03)',
    color: winner ? GOLD_L : '#AAAAAA', fontWeight: winner ? '700' : '400',
    fontSize: winner ? 13 : 12,
  }),
  miniBar: (val, max, winner) => ({
    height: 3, borderRadius: 2, marginTop: 4,
    width: `${Math.max(0, (val / max) * 100)}%`,
    background: winner ? GOLD : 'rgba(201,160,44,0.3)',
    minWidth: val < 0 ? 0 : 2,
  }),
  featureSection: {
    background: '#0D0D0D', border: `1px solid rgba(201,160,44,0.18)`,
    borderRadius: 10, padding: 24, marginBottom: 24,
  },
  featureRow: { display: 'flex', alignItems: 'center', gap: 14, marginBottom: 12 },
  featureName: { fontSize: 12, color: '#AAA', width: 280, flexShrink: 0 },
  barWrap: { flex: 1, background: 'rgba(255,255,255,0.04)', borderRadius: 2, height: 12, overflow: 'hidden' },
  barFill: (pct) => ({
    height: '100%', width: `${pct}%`,
    background: `linear-gradient(90deg, ${GOLD}, ${GOLD_L})`,
    borderRadius: 2, transition: 'width 0.6s ease',
  }),
  pct: { fontSize: 10, color: '#666', width: 38, textAlign: 'right' },
  infoBox: {
    background: '#0D0D0D', border: `1px solid rgba(201,160,44,0.2)`,
    borderLeft: `3px solid ${GOLD}`,
    borderRadius: 8, padding: '16px 20px', fontSize: 12, color: '#777', lineHeight: 1.8,
  },
}

export default function Dashboard() {
  const clf = leaderboard.classification
  const reg = leaderboard.regression
  const fi  = leaderboard.feature_importance
  const maxClf = Math.max(...clf.map(m => m.test_f1))
  const maxReg = Math.max(...reg.map(m => m.test_r2))
  const maxFi  = Math.max(...fi.map(f => f.clf))

  return (
    <div>
      <div style={s.sectionTitle}>Model Performance Dashboard</div>

      <div style={s.statsGrid}>
        {[
          { label: 'Classification F1', value: '0.805', sub: '+0.449 vs dummy' },
          { label: 'ROC-AUC',           value: '0.905', sub: 'XGBoost classifier' },
          { label: 'Regression R²',     value: '0.736', sub: 'GradBoost regressor' },
          { label: 'Median Rate Error', value: '€1,677', sub: 'per night (test set)' },
        ].map(c => (
          <div key={c.label} style={s.statCard}>
            <div style={s.statLabel}>{c.label}</div>
            <div style={s.statValue}>{c.value}</div>
            <div style={s.statSub}>{c.sub}</div>
          </div>
        ))}
      </div>

      <div style={s.grid2}>
        <div style={s.card}>
          <div style={s.cardTitle}>Classification — Test F1-macro</div>
          <table style={s.table}>
            <thead>
              <tr>
                <th style={s.th}>Model</th>
                <th style={s.th}>CV F1</th>
                <th style={s.th}>Test F1</th>
                <th style={s.th}>AUC</th>
              </tr>
            </thead>
            <tbody>
              {clf.map(m => {
                const winner = m.test_f1 === maxClf
                return (
                  <tr key={m.model}>
                    <td style={s.td(winner)}>
                      {m.model}
                      <div style={s.miniBar(m.test_f1, 1, winner)} />
                    </td>
                    <td style={s.td(winner)}>{m.cv_f1.toFixed(3)}</td>
                    <td style={s.td(winner)}>{m.test_f1.toFixed(3)}</td>
                    <td style={s.td(winner)}>{m.test_auc.toFixed(3)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        <div style={s.card}>
          <div style={s.cardTitle}>Regression — Test R²</div>
          <table style={s.table}>
            <thead>
              <tr>
                <th style={s.th}>Model</th>
                <th style={s.th}>CV R²</th>
                <th style={s.th}>Test R²</th>
                <th style={s.th}>MAE (€)</th>
              </tr>
            </thead>
            <tbody>
              {reg.map(m => {
                const winner = m.test_r2 === maxReg
                return (
                  <tr key={m.model}>
                    <td style={s.td(winner)}>
                      {m.model}
                      <div style={s.miniBar(Math.max(0, m.test_r2), 1, winner)} />
                    </td>
                    <td style={s.td(winner)}>{m.cv_r2.toFixed(3)}</td>
                    <td style={s.td(winner)}>{m.test_r2.toFixed(3)}</td>
                    <td style={s.td(winner)}>€{m.test_mae_eur.toLocaleString()}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div style={s.featureSection}>
        <div style={s.cardTitle}>Feature Importance — Top Pricing Signals</div>
        {fi.map(f => (
          <div key={f.feature} style={s.featureRow}>
            <div style={s.featureName}>{f.feature}</div>
            <div style={s.barWrap}>
              <div style={s.barFill((f.clf / maxFi) * 100)} />
            </div>
            <div style={s.pct}>{(f.clf * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>

      <div style={s.infoBox}>
        <span style={{ color: GOLD, fontWeight: 700 }}>Honest evaluation (v2 — 24 hotels, interpretable features): </span>
        XGBoost wins classification (F1=0.805, AUC=0.905); GradBoost wins regression (R²=0.736, MAE=€1,677).
        Both beat the Dummy baseline by a large margin. V2 replaces individual hotel OHE dummies
        (which memorised specific brands) with <em>hotel_tier</em> (1–3) + resort type — generalizable, interpretable,
        and VIF-clean. Preliminary EDA identified Cheval Blanc Randheli and The Brando as IQR extreme
        outliers and removed them. Dataset: 1,440 rows, 24 Lartisien-verified hotels across Europe &amp; worldwide.
        <br /><span style={{ color: GOLD_L }}>Top drivers: room category (44% reg) · hotel tier (13%) · Christmas proximity (11%) · seasonality (month)</span>
      </div>
    </div>
  )
}
