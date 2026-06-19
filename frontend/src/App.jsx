import { useState, useRef, useEffect, useCallback } from 'react';
// eslint-disable-next-line no-unused-vars
import { motion, AnimatePresence } from 'framer-motion';
import {
  UploadCloud, Activity, CheckCircle, AlertTriangle,
  Map as MapIcon, Download, FileText, BarChart3, LogOut, Clock,
  Video, Shield, TrendingDown, Coins, Ruler, Layers, CheckCircle2,
  AlertOctagon, MoveHorizontal
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend
} from 'recharts';
import RoadMap from './components/RoadMap';
import ErrorBoundary from './components/ErrorBoundary';
import PotholeAnalysis from './components/PotholeAnalysis';

// ── Risk Banner ──────────────────────────────────────────────────────────────
function RiskBanner({ riskLevel }) {
  const map = {
    GOOD:     { emoji: '✅', color: '#166534', bg: '#DCFCE7', border: '#BBF7D0' },
    MODERATE: { emoji: '⚠️', color: '#92400E', bg: '#FEF3C7', border: '#FDE68A' },
    POOR:     { emoji: '🚨', color: '#991B1B', bg: '#FEE2E2', border: '#FECACA' },
    CRITICAL: { emoji: '🚨', color: '#7F1D1D', bg: '#FEE2E2', border: '#FCA5A5' },
  };
  const style = map[riskLevel] || map.MODERATE;
  return (
    <div style={{
      textAlign: 'center', margin: '16px 0', padding: '14px',
      borderRadius: '10px', background: style.bg, border: `1px solid ${style.border}`
    }}>
      <span style={{ fontSize: '1.2rem', fontWeight: 700, color: style.color }}>
        {style.emoji} Road Condition: {riskLevel}
      </span>
    </div>
  );
}

// ── Pie chart legend (custom) ─────────────────────────────────────────────────
const PIE_COLORS = ['#D72A34', '#003366', '#637381'];
const PIE_LABELS = ['Alligator Cracks', 'Longitudinal', 'Transverse'];

function PieLegend() {
  return (
    <div className="legend">
      {PIE_LABELS.map((label, i) => (
        <div key={label} className="legend-item">
          <div className="legend-dot" style={{ background: PIE_COLORS[i] }} />
          {label}
        </div>
      ))}
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
function App() {
  const [videoFile, setVideoFile]       = useState(null);
  const [gpsFile, setGpsFile]           = useState(null);
  const [isUploading, setIsUploading]   = useState(false);
  const [isAnalyzing, setIsAnalyzing]   = useState(false);

  const [statusMsg, setStatusMsg]       = useState('');
  const [statusType, setStatusType]     = useState('info'); // 'info' | 'error'
  const [progress, setProgress]         = useState({ current: 0, total: 0 });
  const [isDragging, setIsDragging]     = useState(false);
  const [scoreHistory, setScoreHistory] = useState([]);
  const [widthHistory, setWidthHistory] = useState([]);
  const [currentMetrics, setCurrentMetrics] = useState({ chainage_m: 0, road_width_m: 0, lat: null, lon: null });
  const [finalSummary, setFinalSummary] = useState(null);
  const [imageSrc, setImageSrc]         = useState(null);
  const [geoData, setGeoData]           = useState(null);
  const [viewMode, setViewMode]         = useState('analysis');
  const [history, setHistory]           = useState([]);
  const [user, setUser]                 = useState(null);
  const [authMode, setAuthMode]         = useState('login');
  const [authForm, setAuthForm]         = useState({ username: '', password: '', email: '' });

  const fileInputRef        = useRef(null);
  const wsRef               = useRef(null);
  const historyLastFetchRef = useRef(0);

  // ── Helpers ───────────────────────────────────────────────────────────────

  const setStatus = (msg, type = 'info') => { setStatusMsg(msg); setStatusType(type); };

  const MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024;
  const validateFileSize = (file) => {
    if (file.size > MAX_FILE_SIZE) {
      setStatus(`File too large: ${(file.size / 1024 ** 3).toFixed(2)} GB. Maximum is 5 GB.`, 'error');
      return false;
    }
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['mp4', 'avi', 'mov', 'mkv', 'webm'].includes(ext)) {
      setStatus(`Invalid file type: .${ext}. Accepted: mp4, avi, mov, mkv, webm`, 'error');
      return false;
    }
    return true;
  };

  // ── File Handlers ─────────────────────────────────────────────────────────

  const handleFileChange = (e) => {
    if (e.target.files?.[0]) {
      const f = e.target.files[0];
      if (!validateFileSize(f)) return;
      setVideoFile(f);
      setStatus('');
    }
  };

  const handleGpsChange = (e) => {
    if (e.target.files?.[0]) {
      setGpsFile(e.target.files[0]);
    }
  };

  const handleDragOver  = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false); };
  const handleDrop      = (e) => {
    e.preventDefault(); e.stopPropagation(); setIsDragging(false);
    if (e.dataTransfer.files?.[0]) {
      const f = e.dataTransfer.files[0];
      if (!validateFileSize(f)) return;
      setVideoFile(f);
      setStatus('');
    }
  };

  const handleUploadClick = () => fileInputRef.current.click();

  // ── Upload & WebSocket ────────────────────────────────────────────────────

  const handleStartAnalysis = async () => {
    if (!videoFile) return;
    setIsUploading(true);
    setStatus('Uploading video to backend...');

    const formData = new FormData();
    formData.append('file', videoFile);
    if (gpsFile) formData.append('gps_file', gpsFile);

    try {
      const res  = await fetch('/upload', {
        method: 'POST',
        headers: { Authorization: `Bearer ${user.token}` },
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || data.error || `Upload failed ${res.status}`);

      setIsUploading(false);
      setIsAnalyzing(true);
      connectWebSocket(data.job_id);
    } catch (err) {
      console.error(err);
      setStatus(`Upload Error: ${err.message}`, 'error');
      setIsUploading(false);
    }
  };

  const connectWebSocket = useCallback((id, retries = 3) => {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws    = new WebSocket(`${proto}//${window.location.host}/ws/${id}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'status') {
        setStatus(data.message);
      } else if (data.type === 'progress') {
        setStatus(`Analyzing Frame ${data.current} of ${data.total}`);
        setProgress({ current: data.current, total: data.total });
        if (data.image) setImageSrc(data.image);
        if (data.metrics) {
          setCurrentMetrics(data.metrics);
          setWidthHistory(prev => [...prev.slice(-49), { frame: data.current, width: data.metrics.road_width_m }]);
        }
        if (data.score != null) {
          setScoreHistory(prev => [...prev.slice(-49), { frame: data.current, score: data.score }]);
        }
      } else if (data.type === 'complete') {
        setStatus('Analysis Complete!');
        setFinalSummary(data.summary);
        fetchGeoData(id);
        setIsAnalyzing(false);
        ws.close();
        fetchHistory(true);
      } else if (data.type === 'error') {
        setStatus(`Error: ${data.message}`, 'error');
        setIsAnalyzing(false);
        ws.close();
      }
    };

    ws.onclose = (e) => {
      if (!e.wasClean && retries > 0 && isAnalyzing) {
        const delay = (4 - retries) * 2000;
        setStatus(`Connection lost. Reconnecting in ${delay / 1000}s...`);
        setTimeout(() => connectWebSocket(id, retries - 1), delay);
      }
    };

    ws.onerror = () => setStatus('WebSocket error. Retrying...');
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAnalyzing]);

  // ── Data Fetching ─────────────────────────────────────────────────────────

  const fetchHistory = async (force = false) => {
    if (!user) return;
    const now = Date.now();
    if (!force && now - historyLastFetchRef.current < 30_000) return;
    historyLastFetchRef.current = now;
    try {
      const res  = await fetch('/history', { headers: { Authorization: `Bearer ${user.token}` } });
      if (res.status === 401) return logout();
      const data = await res.json();
      setHistory(data);
    } catch (err) { console.error('Failed to fetch history:', err); }
  };

  const fetchGeoData = async (id) => {
    try {
      const res  = await fetch(`/geojson/${id}`, { headers: { Authorization: `Bearer ${user.token}` } });
      const data = await res.json();
      setGeoData(data);
    } catch (err) { console.error('Failed to fetch GeoJSON:', err); setGeoData(null); }
  };

  const downloadReport = async (jobId, type) => {
    if (!jobId) { alert('Job ID not available.'); return; }
    try {
      const res = await fetch(`/download/${jobId}/${type}`, { headers: { Authorization: `Bearer ${user.token}` } });
      if (res.ok) {
        const blob = await res.blob();
        const url  = window.URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = `Hawkeye_Report_${jobId.slice(0, 8)}.${type}`;
        document.body.appendChild(a);
        a.click(); a.remove();
        window.URL.revokeObjectURL(url);
      } else {
        alert('Report not ready yet or permission denied.');
      }
    } catch (err) { console.error('Download error:', err); }
  };

  // ── Auth ──────────────────────────────────────────────────────────────────

  const handleAuth = async (e) => {
    e.preventDefault();
    const endpoint = authMode === 'login' ? '/token' : '/signup';
    try {
      if (authMode === 'login') {
        const fd = new FormData();
        fd.append('username', authForm.username);
        fd.append('password', authForm.password);
        const res  = await fetch(endpoint, { method: 'POST', body: fd });
        let data = {};
        try { data = await res.json(); } catch { data.detail = `Server Error: ${res.status}`; }
        if (data.access_token) {
          sessionStorage.setItem('hawkeye_token', data.access_token);
          sessionStorage.setItem('hawkeye_user', authForm.username);
          setUser({ username: authForm.username, token: data.access_token });
        } else {
          alert(data.detail || 'Login failed');
        }
      } else {
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(authForm),
        });
        if (res.ok) { setAuthMode('login'); alert('Account created! Please login.'); }
        else {
          let data = {};
          try { data = await res.json(); } catch { data.detail = `Server Error: ${res.status}`; }
          alert(data.detail || 'Signup failed');
        }
      }
    } catch (err) {
      alert(`Auth error: ${err.message}. Ensure backend is running on port 8000.`);
    }
  };

  const logout = () => {
    sessionStorage.removeItem('hawkeye_token');
    sessionStorage.removeItem('hawkeye_user');
    setUser(null); setHistory([]); setFinalSummary(null);
  };

  const resetAnalysis = () => {
    setVideoFile(null); setGpsFile(null); setIsUploading(false); setIsAnalyzing(false);
    setStatus(''); setProgress({ current: 0, total: 0 });
    setImageSrc(null); setFinalSummary(null);
    setScoreHistory([]); setWidthHistory([]);
    if (wsRef.current) wsRef.current.close();
  };

  // ── Effects ───────────────────────────────────────────────────────────────

  useEffect(() => {
    const token = sessionStorage.getItem('hawkeye_token');
    const uname = sessionStorage.getItem('hawkeye_user');
    if (token && uname) setUser({ username: uname, token });
    return () => { if (wsRef.current) wsRef.current.close(); };
  }, []);

  useEffect(() => { if (user) fetchHistory(); }, [user]); // eslint-disable-line

  // ── Derived ───────────────────────────────────────────────────────────────

  const progressPercent = progress.total > 0 ? (progress.current / progress.total) * 100 : 0;

  // Merge score + width history for live chart
  const liveChartData = scoreHistory.map((s, i) => ({ ...s, width: widthHistory[i]?.width }));

  // Safe final scores array: backend might send as finalSummary.scores or not at all
  const finalScores = finalSummary?.scores ?? [];

  // Pie chart data
  let pieData = finalSummary ? [
    { name: 'Alligator',    value: +(finalSummary.cracks?.by_type?.alligator?.avg_pct    || 0).toFixed(2) },
    { name: 'Longitudinal', value: +(finalSummary.cracks?.by_type?.longitudinal?.avg_pct || 0).toFixed(2) },
    { name: 'Transverse',   value: +(finalSummary.cracks?.by_type?.transverse?.avg_pct   || 0).toFixed(2) },
  ] : [];
  
  if (pieData.length > 0 && pieData.every(d => d.value === 0)) {
    pieData = [{ name: 'No Cracks Detected', value: 100 }];
  }

  // ── AUTH SCREEN ───────────────────────────────────────────────────────────

  if (!user) {
    return (
      <div className="auth-container">
        <motion.div
          className="auth-box glass-panel"
          initial={{ opacity: 0, scale: 0.9, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="auth-logo">Hawkeye <span className="color-accent">AI</span></div>
          <p className="auth-subtitle">
            {authMode === 'login' ? 'Login to your control panel' : 'Create an administrative account'}
          </p>

          <form onSubmit={handleAuth}>
            <input
              id="auth-username"
              type="text" placeholder="Username" className="auth-input" required
              value={authForm.username}
              onChange={e => setAuthForm({ ...authForm, username: e.target.value })}
            />
            {authMode === 'signup' && (
              <input
                id="auth-email"
                type="email" placeholder="Email Address" className="auth-input" required
                value={authForm.email}
                onChange={e => setAuthForm({ ...authForm, email: e.target.value })}
              />
            )}
            <input
              id="auth-password"
              type="password" placeholder="Password" className="auth-input" required
              value={authForm.password}
              onChange={e => setAuthForm({ ...authForm, password: e.target.value })}
            />
            <button id="auth-submit" type="submit" className="btn-primary" style={{ width: '100%', marginTop: '12px', justifyContent: 'center' }}>
              {authMode === 'login' ? 'Access Console' : 'Initialize Account'}
            </button>
          </form>

          <p className="auth-toggle">
            {authMode === 'login' ? "Don't have an account?" : 'Already have an account?'}
            <span
              className="auth-toggle-link"
              onClick={() => setAuthMode(authMode === 'login' ? 'signup' : 'login')}
            >
              {authMode === 'login' ? 'Create Account' : 'Login instead'}
            </span>
          </p>
        </motion.div>
      </div>
    );
  }

  // ── MAIN DASHBOARD ────────────────────────────────────────────────────────

  return (
    <div className="container">

      {/* ── Header ── */}
      <header style={{ marginBottom: '36px' }}>
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
        >
          Cognitive Road Intelligence.
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          transition={{ duration: 0.7, delay: 0.2 }}
          style={{ marginTop: '6px' }}
        >
          Automated Dashcam Pipeline for Real-time Infrastructure Evaluation.
        </motion.p>

        <div style={{ marginTop: '24px', display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
          <button
            id="tab-analysis"
            className={`tab-btn ${viewMode === 'analysis' ? 'active' : ''}`}
            onClick={() => setViewMode('analysis')}
          >
            <Activity size={14} style={{ display: 'inline', marginRight: 6 }} />
            Analysis Console
          </button>
          <button
            id="tab-history"
            className={`tab-btn ${viewMode === 'history' ? 'active' : ''}`}
            onClick={() => { setViewMode('history'); fetchHistory(true); }}
          >
            <Clock size={14} style={{ display: 'inline', marginRight: 6 }} />
            Historical Records ({history.length})
          </button>
          <button
            id="tab-pothole"
            className={`tab-btn ${viewMode === 'pothole' ? 'active' : ''}`}
            onClick={() => setViewMode('pothole')}
          >
            <AlertTriangle size={14} style={{ display: 'inline', marginRight: 6 }} />
            Pothole Analysis
          </button>
          <div style={{ flex: 1 }} />
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span>User: <span className="color-accent">{user.username}</span></span>
            <button id="logout-btn" className="tab-btn" onClick={logout} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <LogOut size={13} /> Logout
            </button>
          </div>
        </div>
      </header>

      {/* ── Main Grid ── */}
      <div className="grid grid-2">
        <AnimatePresence mode="popLayout">

          {/* ── Pothole Integration ── */}
          {viewMode === 'pothole' && (
            <PotholeAnalysis key="pothole-module" user={user} setStatusExternal={setStatus} />
          )}

          {/* ── Upload & Controls ── */}
          {viewMode === 'analysis' && !finalSummary && (
            <motion.div
              key="upload-panel"
              className="glass-panel"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, x: -40 }}
              transition={{ duration: 0.4 }}
            >
              <h2 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Activity size={20} color="var(--accent-primary)" />
                Start Analysis
              </h2>

              <div
                id="upload-area"
                className={`upload-area ${videoFile ? 'active' : ''} ${isDragging ? 'active' : ''}`}
                onClick={handleUploadClick}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <input
                  type="file" ref={fileInputRef} style={{ display: 'none' }}
                  accept="video/*" onChange={handleFileChange}
                />
                <UploadCloud className="upload-icon" />
                {videoFile ? (
                  <>
                    <p style={{ color: 'var(--text-main)', fontWeight: 600 }}>{videoFile.name}</p>
                    <p style={{ fontSize: '0.82rem', marginTop: 4 }}>
                      {(videoFile.size / 1024 / 1024).toFixed(1)} MB — ready to analyze
                    </p>
                  </>
                ) : (
                  <p>Drag &amp; drop or click to upload dashcam video.<br />
                    <span style={{ fontSize: '0.8rem', opacity: 0.6 }}>mp4, avi, mov, mkv, webm — max 5 GB</span>
                  </p>
                )}
              </div>

              <div style={{ marginTop: '16px', textAlign: 'center' }}>
                <input
                  type="file"
                  id="gps-upload"
                  accept=".csv,.gpx"
                  style={{ display: 'none' }}
                  onChange={handleGpsChange}
                />
                <button
                  className="btn-secondary"
                  style={{ fontSize: '0.8rem', padding: '6px 12px' }}
                  onClick={() => document.getElementById('gps-upload').click()}
                >
                  <MapIcon size={14} style={{ display: 'inline', marginRight: '6px' }} />
                  {gpsFile ? `GPS: ${gpsFile.name}` : 'Upload GPS / Heartbeat (Optional)'}
                </button>
              </div>

              <div style={{ marginTop: '24px', textAlign: 'center' }}>
                <button
                  id="start-analysis-btn"
                  className="btn-primary"
                  onClick={handleStartAnalysis}
                  disabled={!videoFile || isUploading || isAnalyzing}
                >
                  {isUploading ? '⏳ Uploading...' : isAnalyzing ? '🔄 Analyzing...' : '🚀 Launch Intelligence Pipeline'}
                </button>
              </div>

              {statusMsg && (
                <motion.div
                  key={statusMsg}
                  className={`status-msg ${statusType === 'error' ? 'error' : ''}`}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  {statusMsg}
                </motion.div>
              )}

              {(isUploading || isAnalyzing) && (
                <div className="progress-container">
                  <div className="progress-bar" style={{ width: `${Math.max(progressPercent, 2)}%` }} />
                </div>
              )}
            </motion.div>
          )}

          {/* ── Live Telemetry ── */}
          {viewMode === 'analysis' && (isAnalyzing || imageSrc) && !finalSummary && (
            <motion.div
              key="live-telemetry"
              className="glass-panel"
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4 }}
              style={{ display: 'flex', flexDirection: 'column' }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h2 style={{ margin: 0 }}>Scientific Telemetry</h2>
                {isAnalyzing && <div className="live-badge">HAWKEYE LIVE</div>}
              </div>

              {imageSrc ? (
                <div style={{ position: 'relative' }}>
                  <img src={imageSrc} alt="Live AI Inference" className="live-feed" />
                  <div className="telemetry-overlay">
                    <div>CH: {(currentMetrics.chainage_m || 0).toFixed(3)} km</div>
                    <div>W: {(currentMetrics.road_width_m || 0).toFixed(2)} m</div>
                    {currentMetrics.lat && (
                      <div>{currentMetrics.lat.toFixed(5)}, {currentMetrics.lon?.toFixed(5)}</div>
                    )}
                  </div>
                </div>
              ) : (
                <div style={{ width: '100%', height: '260px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.3)', borderRadius: '10px' }}>
                  <p>Awaiting stream...</p>
                </div>
              )}

              <div className="grid grid-2" style={{ marginTop: '14px', gap: '10px' }}>
                <div className="mini-metric">
                  <span className="label">Chainage</span>
                  <span className="value">{(currentMetrics.chainage_m || 0).toFixed(3)} km</span>
                </div>
                <div className="mini-metric">
                  <span className="label">GPS</span>
                  <span className="value" style={{ fontSize: '0.72rem' }}>
                    {currentMetrics.lat?.toFixed(5) ?? '—'}, {currentMetrics.lon?.toFixed(5) ?? '—'}
                  </span>
                </div>
              </div>

              {liveChartData.length > 1 && (
                <div style={{ marginTop: '20px' }}>
                  <p style={{ fontSize: '0.75rem', opacity: 0.55, marginBottom: '8px' }}>
                    Safety Score (cyan) &amp; Road Width (yellow)
                  </p>
                  <ResponsiveContainer width="100%" height={130} minWidth={0}>
                    <LineChart data={liveChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
                      <XAxis dataKey="frame" hide />
                      <YAxis yAxisId="left"  domain={[0, 100]} hide />
                      <YAxis yAxisId="right" orientation="right" domain={[0, 15]} hide />
                      <Tooltip contentStyle={{ background: '#FFFFFF', border: '1px solid #E5E8EB', borderRadius: 8, color: '#212B36' }} />
                      <Line yAxisId="left"  type="monotone" dataKey="score" stroke="#D72A34" strokeWidth={2} dot={false} isAnimationActive={false} />
                      <Line yAxisId="right" type="monotone" dataKey="width" stroke="#003366" strokeWidth={2} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </motion.div>
          )}

          {/* ── Final Results ── */}
          {viewMode === 'analysis' && finalSummary && (
            <motion.div
              key="final-results"
              className="glass-panel"
              style={{ gridColumn: '1 / -1' }}
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <CheckCircle color="#00e87a" /> Evaluation Summary
              </h2>

              {/* Risk Banner */}
              {finalSummary.overall?.risk_level && (
                <RiskBanner riskLevel={finalSummary.overall.risk_level} />
              )}

              {/* Primary Metrics */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '16px', marginTop: '24px' }}>
                <div className="metric-card">
                  <div className="icon-wrapper"><Video size={20} /></div>
                  <div className="metric-value">{finalSummary.metadata?.total_frames_analyzed ?? '—'}</div>
                  <div className="metric-label">Frames Processed</div>
                </div>
                <div className="metric-card">
                  <div className="icon-wrapper"><Clock size={20} /></div>
                  <div className="metric-value">{finalSummary.metadata?.duration_sec ?? '—'} s</div>
                  <div className="metric-label">Video Duration</div>
                </div>
                <div className="metric-card">
                  <div className="icon-wrapper"><Shield size={20} /></div>
                  <div className={`metric-value ${
                    (finalSummary.overall?.safety_score ?? 0) >= 80 ? 'color-good' :
                    (finalSummary.overall?.safety_score ?? 0) >= 60 ? 'color-mod'  : 'color-crit'
                  }`}>
                    {finalSummary.overall?.safety_score ?? '—'}
                  </div>
                  <div className="metric-label">Avg Safety Score</div>
                </div>
                <div className="metric-card">
                  <div className="icon-wrapper"><AlertOctagon size={20} /></div>
                  <div className={`metric-value ${
                    finalSummary.overall?.risk_level === 'GOOD' ? 'color-good' :
                    finalSummary.overall?.risk_level === 'MODERATE' ? 'color-mod' :
                    finalSummary.overall?.risk_level === 'POOR' ? 'color-poor' : 'color-crit'
                  }`} style={{ fontSize: '1.4rem' }}>
                    {finalSummary.overall?.risk_level ?? '—'}
                  </div>
                  <div className="metric-label">Risk Level</div>
                </div>
                <div className="metric-card">
                  <div className="icon-wrapper"><TrendingDown size={20} /></div>
                  <div className="metric-value color-mod">
                    {(finalSummary.cracks?.total_avg_coverage_pct ?? 0).toFixed(2)}%
                  </div>
                  <div className="metric-label">Avg Crack Cover</div>
                </div>

                <div className="metric-card">
                  <div className="icon-wrapper"><Ruler size={20} /></div>
                  <div className="metric-value" style={{ fontSize: '1.6rem' }}>
                    {finalSummary.overall?.road_width_avg_m ?? '—'} m
                  </div>
                  <div className="metric-label">Avg Road Width</div>
                </div>

                <div className="metric-card">
                  <div className="icon-wrapper"><Layers size={20} /></div>
                  <div className="metric-value" style={{ fontSize: '1.4rem' }}>
                    {finalSummary.overall?.surface_type?.toUpperCase() ?? '—'}
                  </div>
                  <div className="metric-label">Surface Type</div>
                </div>
                <div className="metric-card">
                  <div className="icon-wrapper"><MapIcon size={20} /></div>
                  <div className="metric-value" style={{ fontSize: '1.6rem' }}>
                    {finalSummary.overall?.signboard_coverage_pct ?? 0}%
                  </div>
                  <div className="metric-label">Signboard Cover</div>
                </div>
                <div className="metric-card">
                  <div className="icon-wrapper"><CheckCircle2 size={20} /></div>
                  <div className="metric-value color-good">{finalSummary.score_distribution?.frames_good ?? 0}</div>
                  <div className="metric-label">Good Frames</div>
                </div>
                <div className="metric-card">
                  <div className="icon-wrapper"><AlertTriangle size={20} /></div>
                  <div className="metric-value color-crit">{finalSummary.score_distribution?.frames_critical ?? 0}</div>
                  <div className="metric-label">Critical Frames</div>
                </div>
              </div>

              {/* Spatial Map */}
              <div style={{ marginTop: '36px' }}>
                <h2 style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                  <MapIcon color="var(--accent-primary)" /> Spatial Health Analysis
                </h2>
                <ErrorBoundary fallback={
                  <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)' }}>
                    Map failed to load. GeoJSON may not be available yet.
                  </div>
                }>
                  <RoadMap geoData={geoData} />
                </ErrorBoundary>
              </div>

              {/* Charts row */}
              <div className="grid grid-2" style={{ marginTop: '36px' }}>
                {/* Safety Score Trend */}
                <div className="glass-panel" style={{ height: '340px', display: 'flex', flexDirection: 'column' }}>
                  <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px', flexShrink: 0 }}>
                    <Activity size={16} color="var(--accent-primary)" /> Safety Score Trend
                  </h3>
                  <ErrorBoundary>
                    {finalScores.length > 0 ? (
                      <ResponsiveContainer width="100%" height={240} minWidth={0}>
                        <LineChart data={
                          finalScores.length > 500
                            ? finalScores.filter((_, i) => i % Math.ceil(finalScores.length / 500) === 0)
                            : finalScores
                        }>
                          <CartesianGrid strokeDasharray="3 3" stroke="#E5E8EB" />
                          <XAxis dataKey="frame" hide />
                          <YAxis domain={[0, 100]} stroke="#637381" fontSize={11} />
                          <Tooltip contentStyle={{ background: '#FFFFFF', border: '1px solid #D72A34', borderRadius: 8, color: '#212B36' }} />
                          <Line type="monotone" dataKey="score" stroke="#D72A34" strokeWidth={2.5} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <p style={{ fontSize: '0.85rem', opacity: 0.5 }}>Score trend data not available for this session.</p>
                      </div>
                    )}
                  </ErrorBoundary>
                </div>

                {/* Distress Distribution Pie */}
                <div className="glass-panel" style={{ height: '340px', display: 'flex', flexDirection: 'column' }}>
                  <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px', flexShrink: 0 }}>
                    <BarChart3 size={16} color="var(--accent-primary)" /> Distress Distribution
                  </h3>
                  <ErrorBoundary>
                    <ResponsiveContainer width="100%" height={200} minWidth={0}>
                      <PieChart>
                        <Pie
                          data={pieData}
                          cx="50%" cy="50%"
                          innerRadius={55} outerRadius={78}
                          paddingAngle={4} dataKey="value"
                        >
                          {pieData.map((_, index) => (
                            <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip
                          formatter={(value) => [`${value}%`, '']}
                          contentStyle={{ background: '#FFFFFF', border: '1px solid var(--accent-primary)', borderRadius: 8, color: '#212B36' }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                    <PieLegend />
                  </ErrorBoundary>
                </div>
              </div>

              {/* Export panel */}
              <div style={{ marginTop: '24px' }}>
                <div className="glass-panel" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px', padding: '20px 32px' }}>
                  <div>
                    <h3>Export Administrative Reports</h3>
                    <p style={{ fontSize: '0.87rem', marginTop: 4 }}>
                      Professional reports for administrative records and contractor tenders.
                    </p>
                  </div>
                  <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                    <button
                      id="download-pdf-btn"
                      className="btn-secondary"
                      onClick={() => downloadReport(finalSummary.metadata?.job_id, 'pdf')}
                    >
                      <Download size={16} /> PDF Report
                    </button>
                    <button
                      id="download-csv-btn"
                      className="btn-secondary"
                      onClick={() => downloadReport(finalSummary.metadata?.job_id, 'csv')}
                    >
                      <FileText size={16} /> CSV Data
                    </button>
                  </div>
                </div>
              </div>

              {/* New Analysis button */}
              <div style={{ marginTop: '32px', textAlign: 'center' }}>
                <button id="new-analysis-btn" className="btn-primary" onClick={resetAnalysis}>
                  🔁 New Analysis
                </button>
              </div>
            </motion.div>
          )}

        </AnimatePresence>

        {/* ── History View ── */}
        {viewMode === 'history' && (
          <motion.div
            className="history-grid"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            {history.length === 0 ? (
              <div className="glass-panel" style={{ textAlign: 'center', gridColumn: '1 / -1', padding: '40px' }}>
                <AlertTriangle size={32} color="var(--text-muted)" style={{ marginBottom: 12 }} />
                <p>No records found. Run an analysis to populate the database.</p>
              </div>
            ) : (
              history.map(item => (
                <div
                  key={item.id}
                  id={`history-card-${item.id.slice(0, 8)}`}
                  className="history-card glass-panel"
                  onClick={() => {
                    setFinalSummary(item.full_report_json);
                    fetchGeoData(item.id);
                    setViewMode('analysis');
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <span className="id-tag">{item.id.slice(0, 8)}</span>
                    <span className={`risk-tag ${item.risk_level}`}>{item.risk_level}</span>
                  </div>
                  <h3 style={{ fontSize: '1rem', marginBottom: '10px', lineHeight: 1.4 }}>{item.video_name}</h3>
                  <div className="mini-stats">
                    <span>Score: <span style={{ color: 'var(--text-main)', fontWeight: 600 }}>{item.safety_score?.toFixed(1)}</span></span>
                    <span>Potholes: <span style={{ color: 'var(--text-main)', fontWeight: 600 }}>{item.total_potholes}</span></span>
                  </div>
                  <div className="date-text">{new Date(item.processed_at).toLocaleString()}</div>
                </div>
              ))
            )}
          </motion.div>
        )}
      </div>
    </div>
  );
}

export default App;
