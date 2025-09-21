import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Activity, Cpu, MemoryStick, Server, Bot, FileJson, TestTube2, Home, ShieldCheck, Info, Upload } from 'lucide-react';
import { GlassCard, GlassButton, GlassInput, GlassSelect } from './components/GlassCard';
import DataVisualization from './components/DataVisualization';
import Notification from './components/Notification';
import useNotification from './hooks/useNotification';

// --- Configuration ---
const API_BASE_URL = "http://127.0.0.1:8000";

const ANOMALY_COLORS = {
  normal: '#10B981',
  anomaly: '#EF4444',
  unknown: '#6B7280'
};

const ANOMALY_TYPES = {
  normal: 'Normal',
  anomaly: 'Anomaly',
  unknown: 'Unknown'
};

const navItems = [
    { name: 'Dashboard', icon: Home, section: 'dashboard' },
    { name: 'System Status', icon: Server, section: 'status' },
    { name: 'Loaded Models', icon: Bot, section: 'models' },
    { name: 'Real-time Test', icon: TestTube2, section: 'predict' },
    { name: 'Batch Prediction', icon: FileJson, section: 'batch' },
    { name: 'Simulate Data', icon: Upload, section: 'simulate' },
];

// --- Custom Hooks ---
const useApi = (endpoint, options = {}) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const { showNotification } = useNotification();

    const execute = useCallback(async (body = null, queryParams = {}) => {
        setLoading(true);
        setError(null);
        let url = `${API_BASE_URL}${endpoint}`;
        if (Object.keys(queryParams).length > 0) {
            url += `?${new URLSearchParams(queryParams)}`;
        }
        
        const fetchOptions = {
            method: options.method || 'GET',
            headers: { 'Content-Type': 'application/json', ...options.headers },
        };

        if (body && fetchOptions.method !== 'GET') {
            fetchOptions.body = JSON.stringify(body);
        }

        try {
            const response = await fetch(url, fetchOptions);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }
            const result = await response.json();
            setData(result);
            return result;
        } catch (e) {
            setError(e.message);
            showNotification(e.message, 'error');
            console.error(`API call to ${endpoint} failed:`, e);
            return null;
        } finally {
            setLoading(false);
        }
    }, [endpoint, options.method, options.headers, showNotification]);

    return { data, loading, error, execute };
};

// --- UI Components ---
const Loader = ({ message = "Loading..." }) => (
    <div className="flex flex-col items-center justify-center p-8 text-white">
        <svg className="animate-spin h-8 w-8 text-cyan-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        <p className="mt-4 text-lg">{message}</p>
    </div>
);

const ErrorDisplay = ({ message }) => (
    <GlassCard className="p-6 bg-red-500/20 border-red-500/50">
        <div className="flex items-center">
            <Info className="w-8 h-8 text-red-400 mr-4" />
            <div>
                <h3 className="text-xl font-bold text-red-300">An Error Occurred</h3>
                <p className="text-white mt-1">{message}</p>
            </div>
        </div>
    </GlassCard>
);

const Sidebar = ({ activeSection, setActiveSection }) => {
    return (
        <GlassCard className="h-full p-4 flex flex-col">
            <div className="flex items-center mb-8">
                <ShieldCheck className="text-cyan-300 w-10 h-10 mr-3" />
                <h1 className="text-xl font-bold text-white">WSN Monitor</h1>
            </div>
            <nav className="flex-grow">
                <ul>
                    {navItems.map(item => (
                        <li key={item.name} className="mb-2">
                            <a
                                href={`#${item.section}`}
                                onClick={() => setActiveSection(item.section)}
                                className={`flex items-center p-3 rounded-lg transition-all duration-200 ${activeSection === item.section ? 'bg-cyan-400/30 text-white' : 'text-gray-300 hover:bg-white/10 hover:text-white'}`}
                            >
                                <item.icon className="w-5 h-5 mr-4" />
                                <span>{item.name}</span>
                            </a>
                        </li>
                    ))}
                </ul>
            </nav>
            <div className="text-center text-gray-400 text-xs">
                <p>Version 3.1</p>
                <p>&copy; 2025 WSN Anomaly Detection</p>
            </div>
        </GlassCard>
    );
};

const StatCard = ({ icon, title, value, unit, color }) => {
    const IconComponent = icon;
    return (
        <GlassCard className="p-4 flex-1">
            <div className="flex items-center">
                <div className={`p-3 rounded-full mr-4 ${color}`}>
                    <IconComponent className="w-6 h-6 text-white" />
                </div>
                <div>
                    <p className="text-sm text-gray-300">{title}</p>
                    <p className="text-2xl font-bold text-white">{value} <span className="text-lg">{unit}</span></p>
                </div>
            </div>
        </GlassCard>
    );
};

// --- Main App Component ---
const App = () => {
    const [activeSection, setActiveSection] = useState('dashboard');
    const { notification, showNotification, hideNotification } = useNotification();

    const renderContent = () => {
        switch (activeSection) {
            case 'dashboard':
                return <DashboardSection showNotification={showNotification} />;
            case 'status':
                return <SystemStatusSection />;
            case 'models':
                return <ModelsSection />;
            case 'predict':
                return <RealtimePredictionSection showNotification={showNotification} />;
            case 'batch':
                return <BatchPredictionSection showNotification={showNotification} />;
            case 'simulate':
                return <SimulateDataSection showNotification={showNotification} />;
            default:
                return <DashboardSection showNotification={showNotification} />;
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
            <div className="container mx-auto px-4 py-8">
                <div className="grid grid-cols-12 gap-6">
                    <div className="col-span-12 lg:col-span-3">
                        <Sidebar activeSection={activeSection} setActiveSection={setActiveSection} />
                    </div>
                    <div className="col-span-12 lg:col-span-9">
                        {renderContent()}
                    </div>
                </div>
            </div>
            {notification && (
                <Notification
                    type={notification.type}
                    message={notification.message}
                    onClose={hideNotification}
                />
            )}
        </div>
    );
};

export default App;

// --- Section Components ---
// Note: These components (DashboardSection, SystemStatusSection, etc.) should be
// moved to separate files in a real application for better organization

const DashboardSection = ({ showNotification }) => {
    const { data: rootData, loading: rootLoading, execute: fetchRoot } = useApi('/');
    const { data: healthData, loading: healthLoading, execute: fetchHealth } = useApi('/health');

    useEffect(() => {
        fetchRoot();
        fetchHealth();
        const interval = setInterval(() => {
            fetchRoot();
            fetchHealth();
        }, 30000);
        return () => clearInterval(interval);
    }, [fetchRoot, fetchHealth]);

    return (
        <div>
            <h2 className="text-3xl font-bold text-white mb-6">Dashboard</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <GlassCard className="p-6">
                    <h3 className="text-xl font-semibold text-white mb-4">API Root Status</h3>
                    {rootLoading && <Loader />}

                    {rootData && <p className="text-green-300">{rootData.message}</p>}
                </GlassCard>
                <GlassCard className="p-6">
                    <h3 className="text-xl font-semibold text-white mb-4">API Health Check</h3>
                    {healthLoading && <Loader />}

                    {healthData && (
                        <div>
                            <p className="text-green-300">Status: {healthData.status}</p>
                            <p className="text-gray-300 mt-2">Timestamp: {new Date(healthData.timestamp).toLocaleString()}</p>
                        </div>
                    )}
                </GlassCard>
            </div>
        </div>
    );
};

const SystemStatusSection = () => {
    const { data, loading, error, execute } = useApi('/status');

    useEffect(() => {
        const interval = setInterval(() => execute(), 5000);
        execute();
        return () => clearInterval(interval);
    }, [execute]);

    if (loading && !data) return <Loader message="Fetching system status..." />;
    if (error) return <ErrorDisplay message={error} />;
    if (!data) return null;

    const uptimeSeconds = data.uptime_seconds || 0;
    const hours = Math.floor(uptimeSeconds / 3600);
    const minutes = Math.floor((uptimeSeconds % 3600) / 60);
    const seconds = Math.floor(uptimeSeconds % 60);

    return (
        <div>
            <h2 className="text-3xl font-bold text-white mb-6">System Status</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
                <StatCard icon={Server} title="API Version" value={data.api_version} unit="" color="bg-blue-500" />
                <StatCard icon={Cpu} title="CPU Usage" value={data.cpu_usage_percent?.toFixed(1) || '0'} unit="%" color="bg-green-500" />
                <StatCard icon={MemoryStick} title="Memory Usage" value={data.memory_usage_mb?.toFixed(1) || '0'} unit="MB" color="bg-yellow-500" />
                <StatCard icon={Activity} title="Uptime" value={`${hours}h ${minutes}m ${seconds}s`} unit="" color="bg-purple-500" />
            </div>
            <GlassCard className="p-6">
                <h3 className="text-xl font-semibold text-white mb-4">Model Loading Status</h3>
                <ul className="space-y-2">
                    {data.models_loaded && Object.entries(data.models_loaded).map(([model, loaded]) => (
                        <li key={model} className="flex justify-between items-center text-white">
                            <span>{model}</span>
                            <span className={`px-3 py-1 text-xs font-semibold rounded-full ${loaded ? 'bg-green-500/30 text-green-300' : 'bg-red-500/30 text-red-300'}`}>
                                {loaded ? 'Loaded' : 'Not Loaded'}
                            </span>
                        </li>
                    ))}
                </ul>
            </GlassCard>
        </div>
    );
};

const ModelsSection = () => {
    const { data, loading, error, execute } = useApi('/models');

    useEffect(() => {
        execute();
    }, [execute]);

    if (loading) return <Loader message="Fetching model information..." />;
    if (error) return <ErrorDisplay message={error} />;
    
    return (
        <div>
            <h2 className="text-3xl font-bold text-white mb-6">Loaded Models</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {data && data.length > 0 ? data.map(model => (
                    <GlassCard key={model.model_name} className="p-6">
                        <h3 className="text-xl font-semibold text-white mb-2">{model.model_name}</h3>
                        <p className="text-cyan-300 mb-4">{model.model_type}</p>
                        <h4 className="font-semibold text-white mb-2">Features Used:</h4>
                        <div className="flex flex-wrap gap-2">
                            {model.features_used.map(feature => (
                                <span key={feature} className="bg-white/20 text-xs text-white px-2 py-1 rounded-full">{feature}</span>
                            ))}
                        </div>
                    </GlassCard>
                )) : (
                    <p className="text-gray-300">No models are currently loaded.</p>
                )}
            </div>
        </div>
    );
};

const RealtimePredictionSection = ({ showNotification }) => {
    const { data, loading, error, execute } = useApi('/predict', { method: 'POST' });
    const [formData, setFormData] = useState({ temperature: 25.0, motion: 0, pulse: 70.0 });
    const [chartData, setChartData] = useState([]);

    const handleInputChange = (e) => {
        const { name, value, type } = e.target;
        setFormData(prev => ({ ...prev, [name]: type === 'number' ? parseFloat(value) : parseInt(value) }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const timestamp = new Date().toISOString();
        const payload = {
            data: { ...formData, timestamp, sensor_id: "realtime_test" },
            return_probabilities: true,
            return_features: true,
        };
        const result = await execute(payload);
        if (result) {
            setChartData(prev => [...prev, { ...formData, timestamp, prediction: result.prediction }].slice(-20));
            showNotification(`Prediction: ${result.prediction}`, result.prediction === 'normal' ? 'success' : 'warning');
        }
    };

    return (
        <div>
            <h2 className="text-3xl font-bold text-white mb-6">Real-time Prediction</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <GlassCard className="p-6">
                    <h3 className="text-xl font-semibold text-white mb-4">Sensor Input</h3>
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-1">Temperature (°C)</label>
                            <GlassInput
                                type="number"
                                name="temperature"
                                value={formData.temperature}
                                onChange={handleInputChange}
                                step="0.1"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-1">Motion</label>
                            <GlassSelect
                                name="motion"
                                value={formData.motion}
                                onChange={handleInputChange}
                            >
                                <option value={0}>No Motion</option>
                                <option value={1}>Motion Detected</option>
                            </GlassSelect>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-1">Pulse (BPM)</label>
                            <GlassInput
                                type="number"
                                name="pulse"
                                value={formData.pulse}
                                onChange={handleInputChange}
                                step="0.1"
                            />
                        </div>
                        <GlassButton
                            type="submit"
                            disabled={loading}
                            className="w-full text-white font-bold py-2 px-4"
                        >
                            {loading ? 'Analyzing...' : 'Predict Anomaly'}
                        </GlassButton>
                    </form>
                </GlassCard>

                <div className="space-y-6">
                    <GlassCard className="p-6">
                        <h3 className="text-xl font-semibold text-white mb-4">Prediction Result</h3>
                        {loading && <Loader />}
                        {error && <ErrorDisplay message={error} />}
                        {data && (
                            <div className="space-y-4">
                                <div className={`p-4 rounded-lg border-2 ${data.prediction === 'normal' ? 'border-green-400 bg-green-500/20' : 'border-red-400 bg-red-500/20'}`}>
                                    <p className="text-lg font-bold text-white">Prediction: <span className={data.prediction === 'normal' ? 'text-green-300' : 'text-red-300'}>{data.prediction}</span></p>
                                    <p className="text-white">Confidence: {(data.confidence * 100).toFixed(2)}%</p>
                                </div>
                            </div>
                        )}
                    </GlassCard>

                    {chartData.length > 0 && (
                        <GlassCard className="p-6">
                            <h3 className="text-xl font-semibold text-white mb-4">Sensor Data Trend</h3>
                            <DataVisualization
                                data={chartData}
                                type="line"
                                height={300}
                            />
                        </GlassCard>
                    )}
                </div>
            </div>
        </div>
    );
};

const BatchPredictionSection = ({ showNotification }) => {
    const { data, loading, error, execute } = useApi('/predict/batch', { method: 'POST' });
    const [file, setFile] = useState(null);
    const fileInputRef = useRef();

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleFileDrop = (e) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
        }
    };

    const handleSubmit = async () => {
        if (!file) {
            showNotification("Please select a file first.", "error");
            return;
        }

        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const text = e.target.result;
                let jsonData;
                if (file.type === "application/json") {
                    jsonData = JSON.parse(text);
                } else { // Assume CSV
                    const lines = text.split('\n').filter(line => line.trim() !== '');
                    const headers = lines[0].split(',').map(h => h.trim());
                    jsonData = lines.slice(1).map(line => {
                        const values = line.split(',');
                        return headers.reduce((obj, header, i) => {
                            obj[header] = values[i];
                            return obj;
                        }, {});
                    });
                }

                const result = await execute({ data: jsonData });
                if (result) {
                    showNotification(`Successfully processed ${jsonData.length} records`, "success");
                }
            } catch (err) {
                showNotification(`Error processing file: ${err.message}`, "error");
            }
        };
        reader.readAsText(file);
    };

    return (
        <div>
            <h2 className="text-3xl font-bold text-white mb-6">Batch Prediction</h2>
            <GlassCard className="p-6">
                <div
                    className="border-2 border-dashed border-gray-400 rounded-lg p-8 text-center"
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={handleFileDrop}
                >
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileChange}
                        accept=".csv,.json"
                        className="hidden"
                    />
                    <div className="space-y-4">
                        <Upload className="w-12 h-12 text-gray-400 mx-auto" />
                        <p className="text-lg text-gray-300">
                            Drag and drop your CSV or JSON file here, or{' '}
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                className="text-cyan-400 hover:text-cyan-300"
                            >
                                browse
                            </button>
                        </p>
                        {file && (
                            <p className="text-sm text-gray-400">
                                Selected file: {file.name}
                            </p>
                        )}
                    </div>
                </div>
                <div className="mt-6 flex justify-end">
                    <GlassButton
                        onClick={handleSubmit}
                        disabled={!file || loading}
                        className="text-white font-bold"
                    >
                        {loading ? 'Processing...' : 'Process File'}
                    </GlassButton>
                </div>
            </GlassCard>

            {data && (
                <GlassCard className="mt-6 p-6">
                    <h3 className="text-xl font-semibold text-white mb-4">Results</h3>
                    <div className="space-y-4">
                        <p className="text-gray-300">
                            Processed {data.length} records
                        </p>
                        <DataVisualization
                            data={data}
                            type="line"
                            height={400}
                        />
                    </div>
                </GlassCard>
            )}
        </div>
    );
};

const SimulateDataSection = ({ showNotification }) => {
    const { loading, error, execute } = useApi('/simulate', { method: 'POST' });
    const [config, setConfig] = useState({
        n_days: 7,
        sampling_rate_seconds: 60,
        inject_anomalies: true
    });

    const handleInputChange = (e) => {
        const { name, value, type, checked } = e.target;
        setConfig(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : parseInt(value)
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const result = await execute(config);
        if (result) {
            showNotification("Successfully generated simulation data", "success");
        }
    };

    return (
        <div>
            <h2 className="text-3xl font-bold text-white mb-6">Simulate Data</h2>
            <GlassCard className="p-6">
                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">
                            Number of Days
                        </label>
                        <GlassInput
                            type="number"
                            name="n_days"
                            value={config.n_days}
                            onChange={handleInputChange}
                            min="1"
                            max="30"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">
                            Sampling Rate (seconds)
                        </label>
                        <GlassInput
                            type="number"
                            name="sampling_rate_seconds"
                            value={config.sampling_rate_seconds}
                            onChange={handleInputChange}
                            min="1"
                            max="3600"
                        />
                    </div>
                    <div className="flex items-center">
                        <input
                            type="checkbox"
                            name="inject_anomalies"
                            checked={config.inject_anomalies}
                            onChange={handleInputChange}
                            className="h-4 w-4 text-cyan-400 focus:ring-cyan-400 border-gray-300 rounded"
                        />
                        <label className="ml-2 text-sm text-gray-300">
                            Inject Anomalies
                        </label>
                    </div>
                    <GlassButton
                        type="submit"
                        disabled={loading}
                        className="w-full text-white font-bold py-2 px-4"
                    >
                        {loading ? 'Generating...' : 'Generate Data'}
                    </GlassButton>
                </form>
            </GlassCard>
        </div>
    );
};
