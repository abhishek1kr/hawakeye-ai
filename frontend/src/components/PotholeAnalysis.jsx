import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, Activity, AlertTriangle, Download, Video, CheckCircle } from 'lucide-react';

export default function PotholeAnalysis({ user, setStatusExternal }) {
  const [videoFile, setVideoFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [statusMsg, setStatusMsg] = useState('');
  const [statusType, setStatusType] = useState('info');
  const [progress, setProgress] = useState({ current: 0, total: 100 });
  const [isDragging, setIsDragging] = useState(false);
  const [finalSummary, setFinalSummary] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [jobId, setJobId] = useState(null);
  
  const [confThreshold, setConfThreshold] = useState(0.50);
  const [saveCsv, setSaveCsv] = useState(true);
  const [saveScreenshots, setSaveScreenshots] = useState(true);

  const fileInputRef = useRef(null);
  const wsRef = useRef(null);

  const setStatus = (msg, type = 'info') => {
    setStatusMsg(msg);
    setStatusType(type);
    if (setStatusExternal) setStatusExternal(msg, type);
  };

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

  const handleFileChange = (e) => {
    if (e.target.files?.[0]) {
      const f = e.target.files[0];
      if (!validateFileSize(f)) return;
      setVideoFile(f);
      setStatus('');
    }
  };

  const handleDragOver = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false); };
  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation(); setIsDragging(false);
    if (e.dataTransfer.files?.[0]) {
      const f = e.dataTransfer.files[0];
      if (!validateFileSize(f)) return;
      setVideoFile(f);
      setStatus('');
    }
  };

  const handleStartAnalysis = async () => {
    if (!videoFile) return;
    setIsUploading(true);
    setStatus('Uploading video to backend...');

    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('conf_threshold', confThreshold);
    formData.append('save_csv', saveCsv);
    formData.append('save_screenshots', saveScreenshots);

    try {
      const res = await fetch('/upload_pothole', {
        method: 'POST',
        headers: { Authorization: `Bearer ${user.token}` },
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || data.error || `Upload failed ${res.status}`);

      setJobId(data.job_id);
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
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/${id}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'status') {
        setStatus(data.message);
      } else if (data.type === 'progress') {
        setStatus(data.message || `Processing...`);
        setProgress({ current: data.current, total: data.total });
        if (data.image) setImageSrc(data.image);
      } else if (data.type === 'complete') {
        setStatus('Analysis Complete!');
        setFinalSummary(data.summary);
        setIsAnalyzing(false);
        ws.close();
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
  }, [isAnalyzing]);

  useEffect(() => {
    return () => { if (wsRef.current) wsRef.current.close(); };
  }, []);

  const downloadReport = async (type) => {
    if (!jobId) { alert('Job ID not available.'); return; }
    try {
      const res = await fetch(`/download_pothole/${jobId}/${type}`, { headers: { Authorization: `Bearer ${user.token}` } });
      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Hawkeye_Pothole_${type}_${jobId.slice(0, 8)}.${type === 'video' ? 'mp4' : type}`;
        document.body.appendChild(a);
        a.click(); a.remove();
        window.URL.revokeObjectURL(url);
      } else {
        alert('File not ready yet or permission denied.');
      }
    } catch (err) { console.error('Download error:', err); }
  };

  const progressPercent = progress.total > 0 ? (progress.current / progress.total) * 100 : 0;

  return (
    <motion.div 
      className="grid grid-2" 
      style={{ gridColumn: '1 / -1', width: '100%' }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.4 }}
    >
      <AnimatePresence mode="popLayout">
        {!finalSummary && (
          <motion.div
            key="upload-panel"
            className="glass-panel"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, x: -40 }}
            transition={{ duration: 0.4 }}
          >
            <h2 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <AlertTriangle size={20} color="var(--accent-primary)" />
              Pothole Detection
            </h2>

            <div
              id="upload-area-pothole"
              className={`upload-area ${videoFile ? 'active' : ''} ${isDragging ? 'active' : ''}`}
              onClick={() => fileInputRef.current.click()}
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
                <p>Drag & drop or click to upload dashcam video.<br />
                  <span style={{ fontSize: '0.8rem', opacity: 0.6 }}>mp4, avi, mov, mkv, webm — max 5 GB</span>
                </p>
              )}
            </div>

            <div style={{ marginTop: '20px', background: '#F4F6F8', padding: '16px', borderRadius: '12px', border: '1px solid #E5E8EB' }}>
              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span>Confidence Threshold</span>
                  <span style={{ color: 'var(--accent-primary)' }}>{confThreshold.toFixed(2)}</span>
                </label>
                <input 
                  type="range" 
                  min="0.1" max="1.0" step="0.05" 
                  value={confThreshold} 
                  onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--accent-primary)' }}
                />
              </div>
              <div style={{ display: 'flex', gap: '20px' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                  <input type="checkbox" checked={saveCsv} onChange={(e) => setSaveCsv(e.target.checked)} />
                  Save Detections to CSV
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
                  <input type="checkbox" checked={saveScreenshots} onChange={(e) => setSaveScreenshots(e.target.checked)} />
                  Capture Screenshots
                </label>
              </div>
            </div>

            <div style={{ marginTop: '24px', textAlign: 'center' }}>
              <button
                className="btn-primary"
                onClick={handleStartAnalysis}
                disabled={!videoFile || isUploading || isAnalyzing}
              >
                {isUploading ? '⏳ Uploading...' : isAnalyzing ? '🔄 Analyzing Potholes...' : '🚀 Start Pothole Detection'}
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

        {(isAnalyzing || imageSrc) && !finalSummary && (
          <motion.div
            key="live-telemetry"
            className="glass-panel"
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4 }}
            style={{ display: 'flex', flexDirection: 'column' }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h2 style={{ margin: 0 }}>Live Pothole Detection Feed</h2>
              {isAnalyzing && <div className="live-badge">ANALYZING</div>}
            </div>

            {imageSrc ? (
              <div style={{ position: 'relative' }}>
                <img src={imageSrc} alt="Live AI Inference" className="live-feed" />
              </div>
            ) : (
              <div style={{ width: '100%', height: '260px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#F4F6F8', borderRadius: '10px', color: 'var(--text-muted)' }}>
                <p>Awaiting stream...</p>
              </div>
            )}
          </motion.div>
        )}

        {finalSummary && (
          <motion.div
            key="final-results"
            className="glass-panel"
            style={{ gridColumn: '1 / -1' }}
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <CheckCircle color="#166534" /> Pothole Detection Results
            </h2>

            <div className="grid grid-2" style={{ marginTop: '24px', gap: '20px' }}>
              <div>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                  <Video size={16} color="var(--accent-primary)" /> Output Video
                </h3>
                <video 
                  controls 
                  src={`/download_pothole/${jobId}/video`} 
                  style={{ width: '100%', borderRadius: '10px', background: '#212B36' }} 
                />
              </div>

              <div>
                <h3 style={{ marginBottom: '16px' }}>Summary</h3>
                <div style={{ 
                  background: '#F4F6F8', 
                  padding: '16px', 
                  borderRadius: '10px',
                  border: '1px solid #E5E8EB',
                  whiteSpace: 'pre-wrap',
                  lineHeight: '1.6',
                  color: 'var(--text-main)'
                }}>
                  {finalSummary.summary.replace(/[*#]/g, '').split('\n').filter(line => !line.toLowerCase().includes('detections')).join('\n')}
                </div>
              </div>
            </div>

            <div style={{ marginTop: '24px' }}>
              <div className="glass-panel" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px', padding: '20px 32px' }}>
                <div>
                  <h3>Export Data</h3>
                  <p style={{ fontSize: '0.87rem', marginTop: 4 }}>
                    Download detections and screenshot evidence.
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '12px' }}>
                  {finalSummary.csv_path && (
                    <button className="btn-secondary" onClick={() => downloadReport('csv')}>
                      <Download size={14} style={{ marginRight: 6 }} /> CSV Data
                    </button>
                  )}
                  {finalSummary.zip_path && (
                    <button className="btn-secondary" onClick={() => downloadReport('zip')}>
                      <Download size={14} style={{ marginRight: 6 }} /> Screenshots ZIP
                    </button>
                  )}
                </div>
              </div>
            </div>
            
            <div style={{ marginTop: '24px', textAlign: 'center' }}>
              <button className="btn-primary" onClick={() => {
                setFinalSummary(null);
                setVideoFile(null);
                setImageSrc(null);
                setProgress({ current: 0, total: 100 });
              }}>
                Analyze Another Video
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
