// web_ui/src/App.jsx
import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, StopCircle, Terminal, Image as ImageIcon, Settings, 
  FolderOpen, Cpu, CheckCircle2, AlertCircle, RefreshCw, 
  Sliders, Video, Compass, Trash2, Download
} from 'lucide-react';

const API_BASE = ""; // Relative path works since FastAPI serves us, or fallback to dev port

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [status, setStatus] = useState({ device: 'cpu', cosmos_model_found: false, venv_python: '', running_jobs: 0 });
  const [settings, setSettings] = useState({
    output_dir: './surflife_dataset',
    highway_output_dir: './highway_defect_dataset',
    yolo_dataset_dir: './yolo_urban_auto_dataset',
    epochs: 10,
    batch_size: 8,
    steps: 25,
    box_threshold: 0.20,
    text_threshold: 0.20,
    model_arch: 'yolov8n-seg.pt',
    classes: 'building, road, grass, car, person, tree, edge of road, sidewalk'
  });
  
  // Job list and logging states
  const [jobs, setJobs] = useState([]);
  const [activeJobId, setActiveJobId] = useState(null);
  const [logs, setLogs] = useState([]);
  const consoleEndRef = useRef(null);
  const sseRef = useRef(null);

  // File explorer states
  const [filesData, setFilesData] = useState({ current_dir: '', dirs: [], files: [] });
  const [currentExploreDir, setCurrentExploreDir] = useState('');
  const [selectedMedia, setSelectedMedia] = useState(null);

  // Generator inputs
  const [genTarget, setGenTarget] = useState('swimmer'); // swimmer / shark / highway
  const [genPrompt, setGenPrompt] = useState('');
  const [genCount, setGenCount] = useState(5);
  const [genSteps, setGenSteps] = useState(25);
  const [genBoxThresh, setGenBoxThresh] = useState(0.22);
  const [genTextThresh, setGenTextThresh] = useState(0.22);
  const [genDinoPrompt, setGenDinoPrompt] = useState('');
  const [genNoAnnotate, setGenNoAnnotate] = useState(false);
  
  // Highway specific inputs
  const [hwyDefect, setHwyDefect] = useState('random');
  const [hwyPerspective, setHwyPerspective] = useState('random');
  const [hwyBoxesOnly, setHwyBoxesOnly] = useState(false);

  // YOLO Trainer inputs
  const [trainVideo, setTrainVideo] = useState('videos/flying-toward-diamond-head-from-palolo-valley.mp4');
  const [trainImageDir, setTrainImageDir] = useState('datasets/VisDrone/VisDrone2019-DET-val/images');
  const [trainNumFrames, setTrainNumFrames] = useState(12);
  const [trainNumImages, setTrainNumImages] = useState(12);
  const [trainEpochs, setTrainEpochs] = useState(10);
  const [trainModelArch, setTrainModelArch] = useState('yolov8n-seg.pt');
  const [trainOutputVideo, setTrainOutputVideo] = useState('videos/diamond_head_yolo_seg_realtime.mp4');

  // Video Processor inputs
  const [procVideo, setProcVideo] = useState('videos/flying-toward-diamond-head-from-palolo-valley.mp4');
  const [procEngine, setProcEngine] = useState('yolo');
  const [procModel, setProcModel] = useState('yolo_urban_auto_dataset/runs/urban_diamond_seg/weights/best.pt');
  const [procThresh, setProcThresh] = useState(0.005);
  const [procStride, setProcStride] = useState(1);
  const [procClasses, setProcClasses] = useState('building, road, grass, car, person, tree, edge of road, sidewalk');
  const [procOutputVideo, setProcOutputVideo] = useState('videos/diamond_head_yolo_seg_detected.mp4');

  // Sandbox inputs
  const [sandboxImageFile, setSandboxImageFile] = useState(null);
  const [sandboxImagePath, setSandboxImagePath] = useState('');
  const [sandboxClasses, setSandboxClasses] = useState('building, road, grass, car, person, tree, edge of road, sidewalk');
  const [sandboxBoxThresh, setSandboxBoxThresh] = useState(0.20);
  const [sandboxTextThresh, setSandboxTextThresh] = useState(0.20);
  const [sandboxOverlay, setSandboxOverlay] = useState(null);
  const [sandboxInstances, setSandboxInstances] = useState([]);
  const [sandboxLoading, setSandboxLoading] = useState(false);

  // Synonyms settings states
  const [synonyms, setSynonyms] = useState({});
  const [newSynInputs, setNewSynInputs] = useState({});

  // Load configuration on boot
  useEffect(() => {
    fetchStatus();
    fetchSettings();
    fetchJobs();
    fetchSynonyms();
    exploreDirectory('');

    const interval = setInterval(() => {
      fetchJobs();
      fetchStatus();
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll logs
  useEffect(() => {
    if (consoleEndRef.current) {
      consoleEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  // Clean SSE on unmount
  useEffect(() => {
    return () => {
      if (sseRef.current) sseRef.current.close();
    };
  }, []);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      const data = await res.json();
      setStatus(data);
    } catch (e) {
      console.error("Error fetching API status", e);
    }
  };

  const fetchSettings = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/settings`);
      const data = await res.json();
      setSettings(data);
      // Sync defaults
      setGenBoxThresh(data.box_threshold);
      setGenTextThresh(data.text_threshold);
      setTrainEpochs(data.epochs);
      setTrainModelArch(data.model_arch);
    } catch (e) {
      console.error("Error fetching settings", e);
    }
  };

  const fetchSynonyms = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/settings/synonyms`);
      const data = await res.json();
      setSynonyms(data);
    } catch (e) {
      console.error("Error fetching synonyms mapping", e);
    }
  };

  const saveSettings = async (newSettings) => {
    try {
      const res = await fetch(`${API_BASE}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newSettings)
      });
      const data = await res.json();
      setSettings(data.settings);
      alert("Settings saved successfully!");
    } catch (e) {
      alert("Failed to save settings: " + e.message);
    }
  };

  const saveSynonyms = async (newSyns) => {
    try {
      const res = await fetch(`${API_BASE}/api/settings/synonyms`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newSyns)
      });
      const data = await res.json();
      if (data.status === 'success') {
        setSynonyms(data.synonyms);
        alert("Synonym mappings saved and cached successfully!");
      }
    } catch (e) {
      alert("Failed to save synonyms: " + e.message);
    }
  };

  const fetchJobs = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/jobs`);
      const data = await res.json();
      setJobs(data);
      
      // If active job finishes, update logging display status
      if (activeJobId) {
        const activeJob = data.find(j => j.job_id === activeJobId);
        if (activeJob && activeJob.status !== 'running') {
          // Finished
        }
      }
    } catch (e) {
      console.error("Error fetching job list", e);
    }
  };

  const exploreDirectory = async (dir) => {
    try {
      const url = `${API_BASE}/api/files/browse` + (dir ? `?directory=${encodeURIComponent(dir)}` : '');
      const res = await fetch(url);
      const data = await res.json();
      setFilesData(data);
      setCurrentExploreDir(data.current_dir);
    } catch (e) {
      console.error("Error browsing directory", e);
    }
  };

  const handleStartJob = async (jobType, params) => {
    try {
      const res = await fetch(`${API_BASE}/api/jobs/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_type: jobType, params })
      });
      const data = await res.json();
      if (data.status === 'success') {
        setActiveJobId(data.job_id);
        bindLogsStream(data.job_id);
        fetchJobs();
      } else {
        alert("Failed to trigger job");
      }
    } catch (e) {
      alert("Error starting job: " + e.message);
    }
  };

  const handleCancelJob = async (jobId) => {
    try {
      const res = await fetch(`${API_BASE}/api/jobs/cancel/${jobId}`, { method: 'POST' });
      const data = await res.json();
      if (data.status === 'success') {
        fetchJobs();
        if (jobId === activeJobId) {
          setLogs(prev => [...prev, "\n[Web Interface] Job cancelled by user.\n"]);
        }
      }
    } catch (e) {
      alert("Error cancelling job: " + e.message);
    }
  };

  const bindLogsStream = (jobId) => {
    if (sseRef.current) {
      sseRef.current.close();
    }
    setLogs([]);
    setActiveJobId(jobId);
    
    const source = new EventSource(`${API_BASE}/api/jobs/stream/${jobId}`);
    sseRef.current = source;

    source.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'log') {
        setLogs(prev => [...prev, data.text]);
      } else if (data.type === 'end') {
        setLogs(prev => [...prev, `\n🏁 Job status updated: ${data.status.toUpperCase()} (Code: ${data.exit_code})\n`]);
        source.close();
        fetchJobs();
      } else if (data.type === 'error') {
        setLogs(prev => [...prev, `\n❌ Log Stream Error: ${data.text}\n`]);
        source.close();
      }
    };

    source.onerror = () => {
      source.close();
    };
  };

  // Grounded-SAM Sandbox Testing
  const handleSandboxSubmit = async (e) => {
    e.preventDefault();
    setSandboxLoading(true);
    setSandboxOverlay(null);
    setSandboxInstances([]);

    const formData = new FormData();
    if (sandboxImageFile) {
      formData.append("file", sandboxImageFile);
    } else if (sandboxImagePath) {
      formData.append("image_path", sandboxImagePath);
    } else {
      alert("Please upload a file or specify a local workspace path!");
      setSandboxLoading(false);
      return;
    }
    
    formData.append("classes", sandboxClasses);
    formData.append("box_threshold", sandboxBoxThresh);
    formData.append("text_threshold", sandboxTextThresh);

    try {
      const res = await fetch(`${API_BASE}/api/sandbox/test`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (data.status === 'success') {
        setSandboxOverlay(data.overlay_url);
        setSandboxInstances(data.instances);
      } else {
        alert("Sandbox run failed: " + data.detail);
      }
    } catch (e) {
      alert("Error: " + e.message);
    } finally {
      setSandboxLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="logo">
          <Cpu size={28} style={{ color: '#00d2ff' }} />
          <span>SurfLifeGen UI</span>
        </div>
        <ul className="menu-list">
          <li className={`menu-item ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
            <Sliders size={20} /> Dashboard
          </li>
          <li className={`menu-item ${activeTab === 'synthesizer' ? 'active' : ''}`} onClick={() => setActiveTab('synthesizer')}>
            <ImageIcon size={20} /> Generator
          </li>
          <li className={`menu-item ${activeTab === 'sandbox' ? 'active' : ''}`} onClick={() => setActiveTab('sandbox')}>
            <Compass size={20} /> SAM Sandbox
          </li>
          <li className={`menu-item ${activeTab === 'trainer' ? 'active' : ''}`} onClick={() => setActiveTab('trainer')}>
            <Terminal size={20} /> YOLO Trainer
          </li>
          <li className={`menu-item ${activeTab === 'processor' ? 'active' : ''}`} onClick={() => setActiveTab('processor')}>
            <Video size={20} /> Video Engine
          </li>
          <li className={`menu-item ${activeTab === 'explorer' ? 'active' : ''}`} onClick={() => setActiveTab('explorer')}>
            <FolderOpen size={20} /> Explorer
          </li>
          <li className={`menu-item ${activeTab === 'settings' ? 'active' : ''}`} onClick={() => setActiveTab('settings')}>
            <Settings size={20} /> Config Settings
          </li>
        </ul>

        {/* Live Mini GPU Status */}
        <div style={{ marginTop: 'auto', padding: '16px', background: 'rgba(255,255,255,0.03)', borderRadius: '12px', border: '1px solid var(--card-border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem' }}>
            <Cpu size={16} color="var(--accent-blue)" />
            <span style={{ fontWeight: '600' }}>Apple Silicon GPU</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '6px' }}>
            Device: <span style={{ textTransform: 'uppercase', color: 'var(--accent-blue)', fontWeight: 'bold' }}>{status.device}</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Cosmos Model: <span style={{ color: status.cosmos_model_found ? 'var(--status-success)' : 'var(--status-failed)' }}>
              {status.cosmos_model_found ? 'LOCAL FOUND' : 'NOT FOUND'}
            </span>
          </div>
        </div>
      </aside>

      {/* Main Panel Content */}
      <main className="main-content">
        
        {/* TAB 1: DASHBOARD */}
        {activeTab === 'dashboard' && (
          <div>
            <div style={{ marginBottom: '32px' }}>
              <h1 style={{ fontSize: '2rem', marginBottom: '8px' }}>SurfLifeGen Control Panel</h1>
              <p style={{ color: 'var(--text-secondary)' }}>Automated local synthetic data generation, semantic annotation, and real-time YOLO model training.</p>
            </div>

            <div className="grid-3" style={{ marginBottom: '32px' }}>
              <div className="card">
                <div className="card-title"><Sliders size={20} color="var(--accent-blue)" /> System Status</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <div>Core device: <span className="badge" style={{ color: 'var(--accent-blue)' }}>{status.device.toUpperCase()}</span></div>
                  <div>Active jobs: <span className="badge">{status.running_jobs} running</span></div>
                </div>
              </div>
              <div className="card">
                <div className="card-title"><FolderOpen size={20} color="var(--accent-purple)" /> Workspace Directories</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.85rem' }}>
                  <div>Dataset: <code style={{ color: 'var(--text-secondary)' }}>{settings.output_dir}</code></div>
                  <div>Defects: <code style={{ color: 'var(--text-secondary)' }}>{settings.highway_output_dir}</code></div>
                </div>
              </div>
              <div className="card">
                <div className="card-title"><CheckCircle2 size={20} color="var(--status-success)" /> Quick Launch Actions</div>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button className="btn btn-primary" style={{ padding: '8px 14px', fontSize: '0.85rem' }} onClick={() => setActiveTab('synthesizer')}>Generate Now</button>
                  <button className="btn btn-secondary" style={{ padding: '8px 14px', fontSize: '0.85rem' }} onClick={() => setActiveTab('sandbox')}>Test Image</button>
                </div>
              </div>
            </div>

            {/* Active or Selected Job Console */}
            <div className="card">
              <div className="card-title"><Terminal size={20} color="var(--accent-blue)" /> Active Execution Logs</div>
              {activeJobId ? (
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <span style={{ fontSize: '0.9rem' }}>Streaming logs for Job: <code style={{ color: 'var(--accent-blue)' }}>{activeJobId}</code></span>
                    <button className="btn btn-secondary" style={{ padding: '6px 12px', fontSize: '0.8rem', color: 'var(--status-failed)' }} onClick={() => handleCancelJob(activeJobId)}>
                      Kill Process
                    </button>
                  </div>
                  <div className="console-wrapper">
                    <div className="console-header">
                      <span>CONSOLE LOGS</span>
                      <span className="status-pill running">RUNNING</span>
                    </div>
                    <div className="console-body">
                      {logs.map((log, i) => (
                        <div key={i} className="console-line">{log}</div>
                      ))}
                      <div ref={consoleEndRef} />
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ padding: '40px 0', textPosition: 'center', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                  <Terminal size={48} style={{ opacity: 0.2, marginBottom: '16px' }} />
                  <p>No active pipeline jobs currently streaming.</p>
                  <p style={{ fontSize: '0.85rem', marginTop: '4px' }}>Trigger a job in the tabs to monitor stdout/stderr logs here.</p>
                </div>
              )}
            </div>

            {/* Job History Table */}
            <div className="card" style={{ marginTop: '24px' }}>
              <div className="card-title"><Compass size={20} color="var(--accent-purple)" /> Pipeline Execution History</div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--card-border)', color: 'var(--text-secondary)', textAlign: 'left' }}>
                      <th style={{ padding: '12px' }}>Job ID</th>
                      <th style={{ padding: '12px' }}>Type</th>
                      <th style={{ padding: '12px' }}>Status</th>
                      <th style={{ padding: '12px' }}>Started At</th>
                      <th style={{ padding: '12px' }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {jobs.map(j => (
                      <tr key={j.job_id} style={{ borderBottom: '1px solid rgba(255,255,255,0.02)' }}>
                        <td style={{ padding: '12px', fontFamily: 'var(--font-mono)' }}>{j.job_id}</td>
                        <td style={{ padding: '12px', fontWeight: '500' }}>{j.job_type.replace('_', ' ').toUpperCase()}</td>
                        <td style={{ padding: '12px' }}>
                          <span className={`status-pill ${j.status}`}>
                            {j.status}
                          </span>
                        </td>
                        <td style={{ padding: '12px', color: 'var(--text-muted)' }}>{new Date(j.start_time).toLocaleString()}</td>
                        <td style={{ padding: '12px' }}>
                          <button className="btn btn-secondary" style={{ padding: '4px 8px', fontSize: '0.75rem' }} onClick={() => bindLogsStream(j.job_id)}>
                            View Console
                          </button>
                        </td>
                      </tr>
                    ))}
                    {jobs.length === 0 && (
                      <tr>
                        <td colSpan="5" style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>No historical logs found.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: SYNTHESIZER */}
        {activeTab === 'synthesizer' && (
          <div>
            <div style={{ marginBottom: '32px' }}>
              <h1 style={{ fontSize: '2rem', marginBottom: '8px' }}>Cosmos 3 Omni Synthesizer</h1>
              <p style={{ color: 'var(--text-secondary)' }}>Configure prompts and generate synthetic images featuring customized defect types or targets.</p>
            </div>

            <div className="grid-2">
              {/* Controls */}
              <div className="card">
                <div className="card-title"><Sliders size={20} color="var(--accent-blue)" /> Generation Parameters</div>
                
                <div className="form-group">
                  <label>Generator Class Target</label>
                  <select className="form-control" value={genTarget} onChange={(e) => setGenTarget(e.target.value)}>
                    <option value="swimmer">Swimmer / Beach Aerials</option>
                    <option value="shark">Shark / Submerged Aerials</option>
                    <option value="highway">Highway Defects / Road Wear</option>
                  </select>
                </div>

                {genTarget === 'highway' ? (
                  <>
                    <div className="form-group">
                      <label>Defect Wear Type</label>
                      <select className="form-control" value={hwyDefect} onChange={(e) => setHwyDefect(e.target.value)}>
                        <option value="random">Randomize Defect</option>
                        <option value="alligator_crack">Alligator Pavement Cracking</option>
                        <option value="pothole">Asphalt Pothole</option>
                        <option value="rutting">Wheel Rutting / Indentations</option>
                        <option value="longitudinal_crack">Single Longitudinal Crack</option>
                      </select>
                    </div>
                    <div className="form-group">
                      <label>Camera Perspective</label>
                      <select className="form-control" value={hwyPerspective} onChange={(e) => setHwyPerspective(e.target.value)}>
                        <option value="random">Randomize Perspective</option>
                        <option value="nadir_drone">Nadir Drone Overhead</option>
                        <option value="low_nadir">Low Altitude Nadir</option>
                        <option value="vehicle_surface">Vehicle Mounted Inspection View</option>
                      </select>
                    </div>
                  </>
                ) : null}

                <div className="form-group">
                  <label>Custom Text Prompt (Overrides Random Modular prompts)</label>
                  <textarea 
                    className="form-control" 
                    style={{ height: '80px', resize: 'vertical' }}
                    placeholder={genTarget === 'highway' ? "e.g., Direct overhead nadir photograph of weathered asphalt with deep cracks..." : "e.g., Direct overhead drone photography showing surfers paddling..."}
                    value={genPrompt}
                    onChange={(e) => setGenPrompt(e.target.value)}
                  />
                </div>

                <div className="grid-2">
                  <div className="form-group">
                    <label>Generate Count</label>
                    <input type="number" className="form-control" value={genCount} onChange={(e) => setGenCount(parseInt(e.target.value) || 1)} />
                  </div>
                  <div className="form-group">
                    <label>Diffusion Steps</label>
                    <input type="number" className="form-control" value={genSteps} onChange={(e) => setGenSteps(parseInt(e.target.value) || 25)} />
                  </div>
                </div>

                <div className="grid-2">
                  <div className="form-group">
                    <label>DINO Box Threshold</label>
                    <input type="number" step="0.01" className="form-control" value={genBoxThresh} onChange={(e) => setGenBoxThresh(parseFloat(e.target.value) || 0.2)} />
                  </div>
                  <div className="form-group">
                    <label>DINO Text Threshold</label>
                    <input type="number" step="0.01" className="form-control" value={genTextThresh} onChange={(e) => setGenTextThresh(parseFloat(e.target.value) || 0.2)} />
                  </div>
                </div>

                <div className="form-group">
                  <label>Custom DINO Detection Prompts (Dot Separated)</label>
                  <input type="text" className="form-control" placeholder="e.g., crack . pothole . asphalt defect ." value={genDinoPrompt} onChange={(e) => setGenDinoPrompt(e.target.value)} />
                </div>

                <div style={{ display: 'flex', gap: '15px', alignItems: 'center', marginBottom: '24px' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.85rem' }}>
                    <input type="checkbox" checked={genNoAnnotate} onChange={(e) => setGenNoAnnotate(e.target.checked)} />
                    Skip DINO Bounding Box Annotations
                  </label>
                  {genTarget === 'highway' && (
                    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.85rem' }}>
                      <input type="checkbox" checked={hwyBoxesOnly} onChange={(e) => setHwyBoxesOnly(e.target.checked)} />
                      Coarse Bounding Boxes Only (No SAM Polygons)
                    </label>
                  )}
                </div>

                <button className="btn btn-primary" style={{ width: '100%' }} onClick={() => {
                  if (genTarget === 'highway') {
                    handleStartJob('generate_highway', {
                      defect_type: hwyDefect,
                      perspective: hwyPerspective,
                      output_dir: settings.highway_output_dir,
                      count: genCount,
                      prompt: genPrompt,
                      steps: genSteps,
                      box_threshold: genBoxThresh,
                      text_threshold: genTextThresh,
                      detection_prompt: genDinoPrompt,
                      no_annotate: genNoAnnotate,
                      boxes_only: hwyBoxesOnly
                    });
                  } else {
                    handleStartJob('generate_swimmers_sharks', {
                      target: genTarget,
                      output_dir: settings.output_dir,
                      bulk_count: genCount,
                      prompt: genPrompt,
                      steps: genSteps,
                      box_threshold: genBoxThresh,
                      text_threshold: genTextThresh,
                      detection_prompt: genDinoPrompt,
                      no_annotate: genNoAnnotate
                    });
                  }
                  setActiveTab('dashboard');
                }}>
                  <Play size={18} /> Launch Generator Job
                </button>
              </div>

              {/* View Output Dir */}
              <div className="card">
                <div className="card-title"><FolderOpen size={20} color="var(--accent-purple)" /> Output Gallery Preview</div>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
                  Saved directly to: <code>{genTarget === 'highway' ? settings.highway_output_dir : settings.output_dir}</code>
                </p>
                <button className="btn btn-secondary" onClick={() => {
                  exploreDirectory(genTarget === 'highway' ? settings.highway_output_dir : settings.output_dir);
                  setActiveTab('explorer');
                }}>
                  Browse Generator Directory
                </button>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: SAM SANDBOX */}
        {activeTab === 'sandbox' && (
          <div>
            <div style={{ marginBottom: '32px' }}>
              <h1 style={{ fontSize: '2rem', marginBottom: '8px' }}>Interactive Grounded-SAM Sandbox</h1>
              <p style={{ color: 'var(--text-secondary)' }}>Instantly test DINO labels, classes, and SAM polygon segmentations on an image to preview labels in real time.</p>
            </div>

            <form onSubmit={handleSandboxSubmit} className="sandbox-split">
              {/* Controls */}
              <div className="card">
                <div className="card-title"><Sliders size={20} color="var(--accent-blue)" /> Test Setup</div>
                
                <div className="form-group">
                  <label>Local Image Path (inside workspace)</label>
                  <input 
                    type="text" 
                    className="form-control" 
                    placeholder="e.g. yolo_urban_auto_dataset/raw_annotated/0000001_02999_d_0000005.jpg" 
                    value={sandboxImagePath}
                    onChange={(e) => {
                      setSandboxImagePath(e.target.value);
                      setSandboxImageFile(null); // Clear file upload if text path is filled
                    }}
                  />
                </div>

                <div style={{ margin: '16px 0', textPosition: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>- OR -</div>

                <div className="form-group">
                  <label>Upload Test Image</label>
                  <input 
                    type="file" 
                    className="form-control" 
                    accept="image/*"
                    onChange={(e) => {
                      if (e.target.files[0]) {
                        setSandboxImageFile(e.target.files[0]);
                        setSandboxImagePath(''); // Clear path if file is uploaded
                      }
                    }} 
                  />
                </div>

                <div className="form-group">
                  <label>Segmentation Classes (Comma Separated)</label>
                  <input type="text" className="form-control" value={sandboxClasses} onChange={(e) => setSandboxClasses(e.target.value)} />
                </div>

                <div className="grid-2">
                  <div className="form-group">
                    <label>Box Confidence Thresh</label>
                    <input type="number" step="0.01" className="form-control" value={sandboxBoxThresh} onChange={(e) => setSandboxBoxThresh(parseFloat(e.target.value) || 0.20)} />
                  </div>
                  <div className="form-group">
                    <label>Text Label Match Thresh</label>
                    <input type="number" step="0.01" className="form-control" value={sandboxTextThresh} onChange={(e) => setSandboxTextThresh(parseFloat(e.target.value) || 0.20)} />
                  </div>
                </div>

                <button type="submit" className="btn btn-primary" style={{ width: '100%', marginTop: '16px' }} disabled={sandboxLoading}>
                  {sandboxLoading ? (
                    <>
                      <RefreshCw className="animate-spin" size={18} /> Processing Segmentations...
                    </>
                  ) : (
                    <>
                      <Play size={18} /> Run Grounded-SAM Segmenter
                    </>
                  )}
                </button>
              </div>

              {/* Output Preview */}
              <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
                <div className="card-title"><ImageIcon size={20} color="var(--accent-purple)" /> Segmented Preview Overlay</div>
                
                <div className="preview-container" style={{ flex: 1 }}>
                  {sandboxOverlay ? (
                    <img src={sandboxOverlay} className="preview-image" alt="Segmented Sandbox Result" />
                  ) : (
                    <div className="preview-placeholder">
                      {sandboxLoading ? (
                        <p>Segmenting in-memory on local M-series core...</p>
                      ) : (
                        <>
                          <ImageIcon size={48} style={{ opacity: 0.15, marginBottom: '12px', margin: '0 auto' }} />
                          <p>Upload a file and run detection to render output.</p>
                        </>
                      )}
                    </div>
                  )}
                </div>

                {sandboxInstances.length > 0 && (
                  <div style={{ marginTop: '20px', maxHeight: '150px', overflowY: 'auto' }}>
                    <div style={{ fontWeight: '600', fontSize: '0.85rem', marginBottom: '8px' }}>Detected Class Instances:</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                      {sandboxInstances.map((inst, i) => (
                        <span key={i} className="badge" style={{ color: '#00d2ff', borderColor: 'rgba(0,210,255,0.2)', borderWidth: '1px', borderStyle: 'solid' }}>
                          {inst.class_name.toUpperCase()} ({Math.round(inst.score * 100)}%)
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </form>
          </div>
        )}

        {/* TAB 4: YOLO TRAINER */}
        {activeTab === 'trainer' && (
          <div>
            <div style={{ marginBottom: '32px' }}>
              <h1 style={{ fontSize: '2rem', marginBottom: '8px' }}>YOLOv8-Seg Apple Silicon Auto-Trainer</h1>
              <p style={{ color: 'var(--text-secondary)' }}>Automate frame extraction, zero-shot label conversion, YOLO format export, and PyTorch MPS training.</p>
            </div>

            <div className="grid-2">
              {/* Configuration */}
              <div className="card">
                <div className="card-title"><Sliders size={20} color="var(--accent-blue)" /> Trainer Parameters</div>

                <div className="form-group">
                  <label>Training Input Video File</label>
                  <input type="text" className="form-control" value={trainVideo} onChange={(e) => setTrainVideo(e.target.value)} />
                </div>

                <div className="form-group">
                  <label>VisDrone Dataset Images Directory</label>
                  <input type="text" className="form-control" value={trainImageDir} onChange={(e) => setTrainImageDir(e.target.value)} />
                </div>

                <div className="grid-2">
                  <div className="form-group">
                    <label>Video Frames to Extract</label>
                    <input type="number" className="form-control" value={trainNumFrames} onChange={(e) => setTrainNumFrames(parseInt(e.target.value) || 12)} />
                  </div>
                  <div className="form-group">
                    <label>VisDrone Images to Include</label>
                    <input type="number" className="form-control" value={trainNumImages} onChange={(e) => setTrainNumImages(parseInt(e.target.value) || 12)} />
                  </div>
                </div>

                <div className="grid-2">
                  <div className="form-group">
                    <label>Training Epochs</label>
                    <input type="number" className="form-control" value={trainEpochs} onChange={(e) => setTrainEpochs(parseInt(e.target.value) || 10)} />
                  </div>
                  <div className="form-group">
                    <label>Base Model Architecture</label>
                    <select className="form-control" value={trainModelArch} onChange={(e) => setTrainModelArch(e.target.value)}>
                      <option value="yolov8n-seg.pt">YOLOv8n-Seg (Nano - Fastest)</option>
                      <option value="yolov8s-seg.pt">YOLOv8s-Seg (Small)</option>
                      <option value="yolov8m-seg.pt">YOLOv8m-Seg (Medium)</option>
                    </select>
                  </div>
                </div>

                <div className="form-group">
                  <label>Output Video Destination</label>
                  <input type="text" className="form-control" value={trainOutputVideo} onChange={(e) => setTrainOutputVideo(e.target.value)} />
                </div>

                <button className="btn btn-primary" style={{ width: '100%', marginTop: '10px' }} onClick={() => {
                  handleStartJob('train_yolo', {
                    video: trainVideo,
                    image_dir: trainImageDir,
                    num_frames: trainNumFrames,
                    num_images: trainNumImages,
                    epochs: trainEpochs,
                    model_arch: trainModelArch,
                    output_video: trainOutputVideo,
                    dataset_dir: settings.yolo_dataset_dir
                  });
                  setActiveTab('dashboard');
                }}>
                  <Play size={18} /> Launch YOLO Training Pipeline
                </button>
              </div>

              {/* Outputs info */}
              <div className="card">
                <div className="card-title"><FolderOpen size={20} color="var(--accent-purple)" /> Generated Outputs Info</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', fontSize: '0.9rem' }}>
                  <div>
                    <span style={{ fontWeight: '600' }}>Formatted YOLO Dataset:</span>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                      Exports to: <code>{settings.yolo_dataset_dir}/yolo_formatted</code>
                    </p>
                  </div>
                  <div>
                    <span style={{ fontWeight: '600' }}>Ultralytics Run Logs:</span>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                      Saved inside: <code>{settings.yolo_dataset_dir}/runs/</code>
                    </p>
                  </div>
                  <div>
                    <span style={{ fontWeight: '600' }}>Inference Output Video:</span>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                      Once training completes, plays processed results at: <code>{trainOutputVideo}</code>
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: VIDEO PROCESSOR */}
        {activeTab === 'processor' && (
          <div>
            <div style={{ marginBottom: '32px' }}>
              <h1 style={{ fontSize: '2rem', marginBottom: '8px' }}>Drone Video Segmentation Engine</h1>
              <p style={{ color: 'var(--text-secondary)' }}>Run frame-by-frame multi-class overlays on inspection or flight videos using Grounded-SAM or custom YOLO weights.</p>
            </div>

            <div className="grid-2">
              <div className="card">
                <div className="card-title"><Sliders size={20} color="var(--accent-blue)" /> Video Config</div>

                <div className="form-group">
                  <label>Input Video Path (.mp4, .mov)</label>
                  <input type="text" className="form-control" value={procVideo} onChange={(e) => setProcVideo(e.target.value)} />
                </div>

                <div className="form-group">
                  <label>Inference Engine</label>
                  <select className="form-control" value={procEngine} onChange={(e) => setProcEngine(e.target.value)}>
                    <option value="yolo">Trained YOLOv8-Seg Weights (50+ FPS)</option>
                    <option value="sam">Zero-shot Grounded-SAM (High Precision, ~1 FPS)</option>
                  </select>
                </div>

                {procEngine === 'yolo' ? (
                  <div className="form-group">
                    <label>Trained Model Weights Path (.pt file)</label>
                    <input type="text" className="form-control" value={procModel} onChange={(e) => setProcModel(e.target.value)} />
                  </div>
                ) : null}

                <div className="grid-2">
                  <div className="form-group">
                    <label>Detection Confidence Thresh</label>
                    <input type="number" step="0.001" className="form-control" value={procThresh} onChange={(e) => setProcThresh(parseFloat(e.target.value) || 0.05)} />
                  </div>
                  <div className="form-group">
                    <label>Frame Process Stride</label>
                    <input type="number" className="form-control" value={procStride} onChange={(e) => setProcStride(parseInt(e.target.value) || 1)} />
                  </div>
                </div>

                <div className="form-group">
                  <label>Classes to Segment (Comma Separated)</label>
                  <input type="text" className="form-control" value={procClasses} onChange={(e) => setProcClasses(e.target.value)} />
                </div>

                <div className="form-group">
                  <label>Output Processed Video Destination</label>
                  <input type="text" className="form-control" value={procOutputVideo} onChange={(e) => setProcOutputVideo(e.target.value)} />
                </div>

                <button className="btn btn-primary" style={{ width: '100%', marginTop: '10px' }} onClick={() => {
                  handleStartJob('segment_video', {
                    video: procVideo,
                    engine: procEngine,
                    model: procEngine === 'yolo' ? procModel : null,
                    box_threshold: procThresh,
                    stride: procStride,
                    classes: procClasses,
                    output_video: procOutputVideo
                  });
                  setActiveTab('dashboard');
                }}>
                  <Play size={18} /> Launch Video Processor Job
                </button>
              </div>

              <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
                <div className="card-title"><Video size={20} color="var(--accent-purple)" /> Rendered Video Viewer</div>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
                  If the processed video has finished rendering, browse the <code>videos/</code> directory in the Explorer tab to play or download it.
                </p>
                <button className="btn btn-secondary" onClick={() => {
                  exploreDirectory('videos');
                  setActiveTab('explorer');
                }}>
                  Open Videos Folder in Explorer
                </button>
              </div>
            </div>
          </div>
        )}

        {/* TAB 6: EXPLORER */}
        {activeTab === 'explorer' && (
          <div>
            <div style={{ marginBottom: '32px' }}>
              <h1 style={{ fontSize: '2rem', marginBottom: '8px' }}>Workspace File Explorer</h1>
              <p style={{ color: 'var(--text-secondary)' }}>Browse generated image datasets, annotations, runs history, and render video outputs directly in the browser.</p>
            </div>

            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.95rem' }}>
                  <FolderOpen size={20} color="var(--accent-blue)" />
                  <span>Directory: <code>{currentExploreDir || '.'}</code></span>
                </div>
                {currentExploreDir && (
                  <button className="btn btn-secondary" style={{ padding: '6px 12px', fontSize: '0.8rem' }} onClick={() => {
                    const parts = currentExploreDir.split('/');
                    parts.pop();
                    exploreDirectory(parts.join('/'));
                  }}>
                    ⬆️ Parent Directory
                  </button>
                )}
              </div>

              {/* Explorer Grid splits */}
              <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '30px' }}>
                
                {/* List Files */}
                <div style={{ borderRight: '1px solid var(--card-border)', paddingRight: '20px', maxHeight: '550px', overflowY: 'auto' }}>
                  <div style={{ fontWeight: '600', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '12px', textTransform: 'uppercase' }}>Subdirectories</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '24px' }}>
                    {filesData.dirs.map(d => (
                      <div key={d.path} style={{ cursor: 'pointer', padding: '8px 12px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', fontSize: '0.9rem' }} onClick={() => exploreDirectory(d.path)}>
                        📁 {d.name}
                      </div>
                    ))}
                    {filesData.dirs.length === 0 && <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>No folders found.</span>}
                  </div>

                  <div style={{ fontWeight: '600', fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '12px', textTransform: 'uppercase' }}>Media & Files</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {filesData.files.map(f => (
                      <div 
                        key={f.path} 
                        style={{ 
                          cursor: 'pointer', 
                          padding: '8px 12px', 
                          background: selectedMedia?.path === f.path ? 'rgba(0,210,255,0.08)' : 'rgba(255,255,255,0.01)', 
                          borderRadius: '8px', 
                          fontSize: '0.85rem',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center'
                        }} 
                        onClick={() => setSelectedMedia(f)}
                      >
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {f.is_media ? '🖼️' : '📄'} {f.name}
                        </span>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                          {Math.round(f.size_bytes / 1024)} KB
                        </span>
                      </div>
                    ))}
                    {filesData.files.length === 0 && <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>No files found.</span>}
                  </div>
                </div>

                {/* Inspect/Player Window */}
                <div>
                  {selectedMedia ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                          <h3>{selectedMedia.name}</h3>
                          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Path: <code>{selectedMedia.path}</code></span>
                        </div>
                        <a 
                          href={`${API_BASE}/api/media/${encodeURIComponent(selectedMedia.path)}`} 
                          download
                          className="btn btn-secondary" 
                          style={{ padding: '6px 12px', fontSize: '0.8rem' }}
                        >
                          <Download size={14} /> Download File
                        </a>
                      </div>

                      <div style={{ background: '#000', borderRadius: '12px', overflow: 'hidden', minHeight: '350px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        {selectedMedia.name.toLowerCase().endsWith('.mp4') || selectedMedia.name.toLowerCase().endsWith('.mov') ? (
                          <video 
                            src={`${API_BASE}/api/media/${encodeURIComponent(selectedMedia.path)}`} 
                            controls 
                            style={{ width: '100%', maxHeight: '450px' }} 
                          />
                        ) : (selectedMedia.name.toLowerCase().endsWith('.png') || selectedMedia.name.toLowerCase().endsWith('.jpg') || selectedMedia.name.toLowerCase().endsWith('.jpeg') || selectedMedia.name.toLowerCase().endsWith('.webp') ? (
                          <img 
                            src={`${API_BASE}/api/media/${encodeURIComponent(selectedMedia.path)}`} 
                            style={{ maxWidth: '100%', maxHeight: '450px', objectFit: 'contain' }} 
                            alt={selectedMedia.name} 
                          />
                        ) : (
                          <div style={{ padding: '40px', color: 'var(--text-secondary)', textAlign: 'center' }}>
                            <p>Cannot render raw file content directly in the browser.</p>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '6px' }}>Please download the file using the button above to view it.</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div style={{ height: '350px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                      <FolderOpen size={48} style={{ opacity: 0.15, marginBottom: '16px' }} />
                      <p>Select a file from the list to preview or stream.</p>
                    </div>
                  )}
                </div>

              </div>
            </div>
          </div>
        )}

        {/* TAB 7: SETTINGS */}
        {activeTab === 'settings' && (
          <div>
            <div style={{ marginBottom: '32px' }}>
              <h1 style={{ fontSize: '2rem', marginBottom: '8px' }}>Global Settings & Taxonomy Config</h1>
              <p style={{ color: 'var(--text-secondary)' }}>Manage path parameters and DINO-to-SAM class synonyms map dictionary.</p>
            </div>

            <div className="grid-2">
              {/* Left Column: Folders */}
              <div className="card">
                <div className="card-title"><Settings size={20} color="var(--accent-blue)" /> System Folders</div>
                
                <div className="form-group">
                  <label>Default Swimmers/Sharks Output Dir</label>
                  <input type="text" className="form-control" value={settings.output_dir} onChange={(e) => setSettings({...settings, output_dir: e.target.value})} />
                </div>

                <div className="form-group">
                  <label>Default Highway Wear Output Dir</label>
                  <input type="text" className="form-control" value={settings.highway_output_dir} onChange={(e) => setSettings({...settings, highway_output_dir: e.target.value})} />
                </div>

                <div className="form-group">
                  <label>Default YOLO Training Dataset Dir</label>
                  <input type="text" className="form-control" value={settings.yolo_dataset_dir} onChange={(e) => setSettings({...settings, yolo_dataset_dir: e.target.value})} />
                </div>

                <div className="grid-2">
                  <div className="form-group">
                    <label>Default Training Epochs</label>
                    <input type="number" className="form-control" value={settings.epochs} onChange={(e) => setSettings({...settings, epochs: parseInt(e.target.value) || 10})} />
                  </div>
                  <div className="form-group">
                    <label>Default Training Batch Size</label>
                    <input type="number" className="form-control" value={settings.batch_size} onChange={(e) => setSettings({...settings, batch_size: parseInt(e.target.value) || 8})} />
                  </div>
                </div>

                <div className="form-group">
                  <label>Default Diffusion Steps</label>
                  <input type="number" className="form-control" value={settings.steps} onChange={(e) => setSettings({...settings, steps: parseInt(e.target.value) || 25})} />
                </div>

                <div className="form-group">
                  <label>Default Segmentation Classes</label>
                  <input type="text" className="form-control" value={settings.classes} onChange={(e) => setSettings({...settings, classes: e.target.value})} />
                </div>

                <button className="btn btn-primary" style={{ width: '100%', marginTop: '16px' }} onClick={() => saveSettings(settings)}>
                  Save Config Settings
                </button>
              </div>

              {/* Right Column: Synonym Editor */}
              <div className="card" style={{ display: 'flex', flexDirection: 'column', maxHeight: '720px', overflowY: 'auto' }}>
                <div className="card-title"><Sliders size={20} color="var(--accent-purple)" /> DINO Class Synonym Mappings</div>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
                  Map raw Grounding DINO text detection queries (e.g. "rooftop", "asphalt") to your target YOLO dataset categories.
                </p>
                
                {Object.keys(synonyms).map(clsName => (
                  <div key={clsName} style={{ marginBottom: '18px', paddingBottom: '10px', borderBottom: '1px solid var(--card-border)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span style={{ fontWeight: '600', fontSize: '0.85rem', color: 'var(--accent-blue)' }}>{clsName.toUpperCase()}</span>
                      <div style={{ display: 'flex', gap: '6px' }}>
                        <input 
                          type="text" 
                          placeholder="Add keyword..." 
                          className="form-control" 
                          style={{ padding: '4px 8px', fontSize: '0.75rem', width: '120px' }}
                          value={newSynInputs[clsName] || ''}
                          onChange={(e) => setNewSynInputs({...newSynInputs, [clsName]: e.target.value})}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              e.preventDefault();
                              const val = newSynInputs[clsName]?.trim().toLowerCase();
                              if (val && !synonyms[clsName].includes(val)) {
                                const updated = { ...synonyms, [clsName]: [...synonyms[clsName], val] };
                                setSynonyms(updated);
                                setNewSynInputs({...newSynInputs, [clsName]: ''});
                              }
                            }
                          }}
                        />
                        <button 
                          className="btn btn-secondary" 
                          style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                          onClick={() => {
                            const val = newSynInputs[clsName]?.trim().toLowerCase();
                            if (val && !synonyms[clsName].includes(val)) {
                              const updated = { ...synonyms, [clsName]: [...synonyms[clsName], val] };
                              setSynonyms(updated);
                              setNewSynInputs({...newSynInputs, [clsName]: ''});
                            }
                          }}
                        >
                          Add
                        </button>
                      </div>
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                      {synonyms[clsName].map(keyword => (
                        <span 
                          key={keyword} 
                          className="badge" 
                          style={{ 
                            display: 'inline-flex', 
                            alignItems: 'center', 
                            gap: '4px',
                            background: 'rgba(255,255,255,0.03)',
                            border: '1px solid var(--card-border)'
                          }}
                        >
                          {keyword}
                          <button 
                            type="button"
                            style={{ background: 'none', border: 'none', color: 'var(--status-failed)', cursor: 'pointer', fontSize: '0.85rem', fontWeight: 'bold' }}
                            onClick={() => {
                              const filtered = synonyms[clsName].filter(k => k !== keyword);
                              setSynonyms({ ...synonyms, [clsName]: filtered });
                            }}
                          >
                            ×
                          </button>
                        </span>
                      ))}
                    </div>
                  </div>
                ))}

                <button className="btn btn-primary" style={{ width: '100%', marginTop: 'auto' }} onClick={() => saveSynonyms(synonyms)}>
                  Save Synonym Mappings
                </button>
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}

export default App;
