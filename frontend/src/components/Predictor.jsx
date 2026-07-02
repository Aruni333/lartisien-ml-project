import { useState } from 'react'
import predictions from '../data/predictions.json'

const HOTELS = [
  // ── Europe: Slovenia neighbours ──────────────────────────────────────────
  { label: 'Aman Venice (Italy)',                     id: 'aman_venice' },
  { label: 'Cipriani Venice — Belmond (Italy)',       id: 'cipriani_venice' },
  { label: 'Four Seasons Hotel Firenze (Italy)',      id: 'four_seasons_firenze' },
  { label: 'Il Sereno Lago di Como (Italy)',          id: 'il_sereno_lago_di_como' },
  { label: 'Le Sirenuse, Positano (Italy)',           id: 'le_sirenuse' },
  { label: 'Rosewood Schloss Fuschl (Austria)',       id: 'rosewood_schloss_fuschl' },
  { label: 'Tennerhof Gourmet Hotel, Kitzbühel (Austria)', id: 'tennerhof_kitzbuhel' },
  { label: 'Four Seasons Gresham Palace, Budapest',  id: 'four_seasons_budapest' },
  // ── Europe: Other luxury ─────────────────────────────────────────────────
  { label: "Badrutt's Palace Hotel, St. Moritz",     id: 'badrutts_palace' },
  { label: 'The Alpina Gstaad (Switzerland)',         id: 'the_alpina_gstaad' },
  { label: 'Airelles Val d\'Isère (France)',          id: 'airelles_val_disere' },
  { label: 'Château de la Messardière, St-Tropez',   id: 'chateau_messardiere' },
  { label: 'Hôtel de Paris Monte-Carlo (Monaco)',     id: 'hotel_de_paris_mc' },
  { label: 'Four Seasons Istanbul at Sultanahmet',   id: 'four_seasons_istanbul' },
  { label: 'Bill & Coo Mykonos (Greece)',             id: 'bill_and_coo_mykonos' },
  { label: 'Canaves Oia Suites, Santorini (Greece)', id: 'canaves_oia_santorini' },
  { label: 'Marbella Club Hotel (Spain)',             id: 'marbella_club' },
  { label: 'Amanzoe, Porto Heli (Greece)',            id: 'amanzoe' },
  // ── International ────────────────────────────────────────────────────────
  { label: 'Bulgari Resort Bali (Indonesia)',         id: 'bulgari_resort_bali' },
  { label: 'Singita Grumeti (Tanzania)',              id: 'singita_grumeti' },
  { label: 'Singita Ebony Lodge (South Africa)',      id: 'singita_ebony_lodge' },
  { label: 'Aman Tokyo (Japan)',                      id: 'aman_tokyo' },
  { label: 'Four Seasons Bora Bora',                  id: 'four_seasons_bora_bora' },
  { label: 'Eden Rock St Barths',                     id: 'eden_rock_st_barths' },
]
const ROOMS  = ['Deluxe Room', 'Superior Room', 'Junior Suite', 'Suite', 'Grand Suite', 'Presidential Suite']
const MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December']

const GOLD   = '#C9A02C'
const GOLD_L = '#E8C96A'
const GOLD_P = 'rgba(201,160,44,0.08)'

const API_URL = import.meta.env.VITE_API_URL || ''

function lookup(hotelId, month, roomType) {
  return predictions.find(p => p.hotel_id === hotelId && p.month === month && p.room_type === roomType)
}

const s = {
  sectionTitle: {
    fontSize: 11, color: GOLD, letterSpacing: '0.22em', textTransform: 'uppercase',
    marginBottom: 24, paddingBottom: 10, borderBottom: `1px solid rgba(201,160,44,0.2)`,
    fontWeight: 600,
  },
  grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px,1fr))', gap: 20, marginBottom: 28 },
  label: {
    display: 'block', color: GOLD, fontSize: 10, letterSpacing: '0.2em',
    textTransform: 'uppercase', marginBottom: 8, fontWeight: '600',
  },
  select: {
    width: '100%', padding: '13px 14px',
    background: '#0D0D0D', border: `1px solid rgba(201,160,44,0.3)`,
    borderRadius: 6, color: '#E0E0E0', fontSize: 14, outline: 'none', appearance: 'none',
    backgroundImage: 'url("data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'12\' height=\'8\' viewBox=\'0 0 12 8\'%3E%3Cpath fill=\'%23C9A02C\' d=\'M6 8L0 0h12z\'/%3E%3C/svg%3E")',
    backgroundRepeat: 'no-repeat', backgroundPosition: 'right 14px center', cursor: 'pointer',
  },
  btn: {
    width: '100%', padding: '16px', cursor: 'pointer',
    background: `linear-gradient(135deg, ${GOLD}, #A07020)`,
    border: 'none', borderRadius: 6, color: '#000', fontSize: 13, fontWeight: '700',
    letterSpacing: '0.2em', textTransform: 'uppercase', marginBottom: 36,
    transition: 'opacity 0.2s', boxShadow: `0 2px 20px rgba(201,160,44,0.25)`,
  },
  results: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 24 },
  card: (premium) => ({
    background: premium ? 'rgba(201,160,44,0.07)' : '#0D0D0D',
    border: `1px solid ${premium ? 'rgba(201,160,44,0.45)' : 'rgba(201,160,44,0.18)'}`,
    borderRadius: 10, padding: 28, textAlign: 'center',
  }),
  cardIcon: { fontSize: 32, marginBottom: 10, display: 'block' },
  cardLabel: { fontSize: 10, letterSpacing: '0.2em', textTransform: 'uppercase', color: '#666', marginBottom: 8 },
  cardValue: (premium) => ({
    fontSize: 22, fontWeight: '700', color: premium ? GOLD_L : GOLD,
  }),
  rateValue: { fontSize: 36, fontWeight: '700', color: GOLD, lineHeight: 1.1 },
  metaGrid: { display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 12, marginBottom: 24 },
  metaCard: {
    background: '#0D0D0D', border: `1px solid rgba(201,160,44,0.15)`,
    borderRadius: 8, padding: '16px', textAlign: 'center',
  },
  metaLabel: { fontSize: 9, letterSpacing: '0.18em', textTransform: 'uppercase', color: '#555', marginBottom: 6 },
  metaValue: { fontSize: 18, fontWeight: '700', color: GOLD },
  disclaimer: {
    background: '#0D0D0D', border: `1px solid rgba(201,160,44,0.15)`,
    borderRadius: 8, padding: '14px 18px', fontSize: 12, color: '#666', lineHeight: 1.7,
    borderLeft: `3px solid rgba(201,160,44,0.4)`,
  },
  probBar: { background: 'rgba(255,255,255,0.06)', borderRadius: 3, height: 6, marginTop: 10, overflow: 'hidden' },
  probFill: (pct) => ({
    height: '100%', width: `${pct}%`, borderRadius: 3, transition: 'width 0.7s ease',
    background: `linear-gradient(90deg, ${GOLD}, ${GOLD_L})`,
  }),
}

export default function Predictor() {
  const [hotel, setHotel] = useState(HOTELS[0].id)
  const [month, setMonth] = useState(7)
  const [room, setRoom]   = useState('Suite')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  async function handlePredict() {
    setLoading(true)
    try {
      if (API_URL) {
        const res = await fetch(`${API_URL}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ hotel_id: hotel, month, room_type: room }),
        })
        if (res.ok) { setResult(await res.json()); setLoading(false); return }
      }
    } catch (_) {}
    const r = lookup(hotel, month, room)
    if (r) setResult({
      is_premium: r.is_premium,
      premium_probability: r.premium_probability,
      estimated_rate_eur: r.estimated_rate_eur,
      label: r.is_premium ? 'Premium (>€3,000)' : 'Standard (≤€3,000)',
      confidence: r.premium_probability > 0.8 || r.premium_probability < 0.2 ? 'High'
                : r.premium_probability > 0.65 || r.premium_probability < 0.35 ? 'Medium' : 'Low',
    })
    setLoading(false)
  }

  const pct = result ? Math.round(result.premium_probability * 100) : 0

  return (
    <div>
      <div style={s.sectionTitle}>Rate Predictor — Select Hotel & Stay Details</div>

      <div style={s.grid}>
        <div>
          <label style={s.label}>Hotel Property</label>
          <select style={s.select} value={hotel} onChange={e => { setHotel(e.target.value); setResult(null) }}>
            {HOTELS.map(h => <option key={h.id} value={h.id}>{h.label}</option>)}
          </select>
        </div>
        <div>
          <label style={s.label}>Month of Stay</label>
          <select style={s.select} value={month} onChange={e => { setMonth(+e.target.value); setResult(null) }}>
            {MONTHS.map((m, i) => <option key={i+1} value={i+1}>{m}</option>)}
          </select>
        </div>
        <div>
          <label style={s.label}>Room Category</label>
          <select style={s.select} value={room} onChange={e => { setRoom(e.target.value); setResult(null) }}>
            {ROOMS.map(r => <option key={r} value={r}>{r}</option>)}
          </select>
        </div>
      </div>

      <button style={s.btn} onClick={handlePredict} disabled={loading}>
        {loading ? '◌  Computing...' : '◆  Predict Rate & Tier'}
      </button>

      {result && (
        <>
          <div style={{ ...s.sectionTitle, marginTop: 0 }}>Prediction Results</div>

          <div style={s.results}>
            <div style={s.card(result.is_premium)}>
              <span style={s.cardIcon}>{result.is_premium ? '◆' : '◇'}</span>
              <div style={s.cardLabel}>Booking Tier</div>
              <div style={s.cardValue(result.is_premium)}>{result.label}</div>
              <div style={s.probBar}>
                <div style={s.probFill(pct)} />
              </div>
              <div style={{ fontSize: 11, color: '#555', marginTop: 8, letterSpacing: '0.05em' }}>
                Premium probability: {pct}% · Confidence: {result.confidence}
              </div>
            </div>

            <div style={s.card(false)}>
              <span style={s.cardIcon}>€</span>
              <div style={s.cardLabel}>Estimated Nightly Rate</div>
              <div style={s.rateValue}>€{result.estimated_rate_eur.toLocaleString()}</div>
              <div style={{ fontSize: 11, color: '#555', marginTop: 10, letterSpacing: '0.08em' }}>per night · EUR</div>
            </div>
          </div>

          <div style={s.metaGrid}>
            {[
              { label: 'Classification F1', value: '0.805' },
              { label: 'Rate Accuracy (R²)', value: '0.736' },
              { label: 'Rate Error (MAE)',   value: '€1,677/night' },
            ].map(c => (
              <div key={c.label} style={s.metaCard}>
                <div style={s.metaLabel}>{c.label}</div>
                <div style={s.metaValue}>{c.value}</div>
              </div>
            ))}
          </div>

          <div style={s.disclaimer}>
            Valid for these 24 Lartisien Collection hotels. Does not capture real-time promotions,
            last-minute rates, or year-over-year price inflation.
            Predictions based on seasonal patterns and room/hotel tier. Presidential Suite carries highest uncertainty.
          </div>
        </>
      )}
    </div>
  )
}
