import React, { useState, useEffect, useCallback, useRef } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts';
import { Upload, Thermometer, Activity, HeartPulse, Cpu, MemoryStick, Server, Bot, FileJson, TestTube2, Home, ShieldCheck, Info, Sparkles } from 'lucide-react';

// --- Configuration ---
const API_BASE_URL = "http://127.0.0.1:8000"; // IMPORTANT: Replace with your actual backend URL if different

const ANOMALY_COLORS = {
    normal: "#2ECC71",
    DoS: "#E74C3C",
    Jamming: "#F39C12",
    Tampering: "#9B59B6",
    HardwareFault: "#3498DB",
    EnvironmentalNoise: "#1ABC9C",
    unknown: "#95A5A6",
    anomaly: "#E74C3C",
    error: "#F1C40F"
};

const ANOMALY_TYPES = {
    0: "normal",
    1: "DoS",
    2: "Jamming",
    3: "Tampering",
    4: "HardwareFault",
    5: "EnvironmentalNoise"
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
            console.error(`API call to ${endpoint} failed:`, e);
            return null;
        } finally {
            setLoading(false);
        }
    }, [endpoint, options.method, options.headers]);

    return { data, loading, error, execute };
};

// New hook for Gemini API calls
const useGemini = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const execute = useCallback(async (prompt) => {
        setLoading(true);
        setError(null);
        setData(null);

        const apiKey = "AIzaSyCGZ5lwEVHTjk39u_4itpXhBOU1WUYsnK0"; // Leave blank, will be handled by the environment
        const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`;

        const payload = {
            contents: [{
                parts: [{ text: prompt }]
            }]
        };

        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorBody = await response.json();
                throw new Error(errorBody.error?.message || 'Gemini API request failed');
            }

            const result = await response.json();
            
            if (result.candidates && result.candidates.length > 0 &&
                result.candidates[0].content && result.candidates[0].content.parts &&
                result.candidates[0].content.parts.length > 0) {
                const text = result.candidates[0].content.parts[0].text;
                setData(text);
                return text;
            } else {
                throw new Error("Invalid response structure from Gemini API");
            }
        } catch (e) {
            setError(e.message);
            console.error("Gemini API call failed:", e);
            return null;
        } finally {
            setLoading(false);
        }
    }, []);

    return { data, loading, error, execute };
};


// --- UI Components ---

const GlassCard = ({ children, className = '', onClick }) => (
    <div 
        className={`bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 shadow-lg transition-all duration-300 hover:shadow-xl hover:bg-white/20 ${className}`}
        onClick={onClick}
    >
        {children}
    </div>
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
                <p>Version 4.0 (Gemini)</p>
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

const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        return (
            <GlassCard className="p-3 text-sm">
                <p className="label text-white font-bold">{`Time: ${label}`}</p>
                {payload.map((p, i) => (
                    <p key={i} style={{ color: p.color }}>{`${p.name}: ${p.value.toFixed(2)}`}</p>
                ))}
            </GlassCard>
        );
    }
    return null;
};

// --- New Gemini Component ---
const GeminiAnalysisCard = ({ prediction, sensorData }) => {
    const { data: analysis, loading, error, execute: getAnalysis } = useGemini();

    const handleAnalysisClick = () => {
        const prompt = `
            You are an expert analyst for a Wireless Sensor Network.
            An anomaly has been detected with the following details:
            - Anomaly Type: ${prediction.prediction}
            - Confidence: ${(prediction.confidence * 100).toFixed(2)}%
            - Temperature: ${sensorData.temperature.toFixed(1)} °C
            - Motion: ${sensorData.motion === 1 ? 'Detected' : 'No Motion'}
            - Pulse: ${sensorData.pulse.toFixed(1)} BPM

            Based on this data, please provide the following in a clear, concise format:
            1. A simple, one-sentence explanation of what this anomaly means in the context of a wireless sensor network.
            2. A numbered list of 3 recommended, actionable steps for a network operator to take to investigate and resolve this issue.
        `;
        getAnalysis(prompt);
    };

    return (
        <GlassCard className="p-6 mt-6 bg-purple-500/10 border-purple-400/50">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                <Sparkles className="w-6 h-6 mr-3 text-purple-300" />
                AI Analysis
            </h3>
            {!analysis && !loading && (
                 <button onClick={handleAnalysisClick} className="w-full flex items-center justify-center bg-purple-500 text-white font-bold py-2 px-4 rounded-lg hover:bg-purple-600 transition duration-300">
                    <Sparkles className="w-5 h-5 mr-2" />
                    Get AI Analysis & Recommendations
                </button>
            )}
            {loading && <Loader message="Getting insights from Gemini..." />}
            {error && <ErrorDisplay message={error} />}
            {analysis && (
                <div className="text-white space-y-3 whitespace-pre-wrap font-mono">
                    {analysis}
                </div>
            )}
        </GlassCard>
    );
};


// --- Page/Section Components ---

const DashboardSection = () => {
    const { data: rootData, loading: rootLoading, error: rootError, execute: fetchRoot } = useApi('/');
    const { data: healthData, loading: healthLoading, error: healthError, execute: fetchHealth } = useApi('/health');

    useEffect(() => {
        fetchRoot();
        fetchHealth();
    }, [fetchRoot, fetchHealth]);

    return (
        <div>
            <h2 className="text-3xl font-bold text-white mb-6">Dashboard</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <GlassCard className="p-6">
                    <h3 className="text-xl font-semibold text-white mb-4">API Root Status</h3>
                    {rootLoading && <Loader />}
                    {rootError && <ErrorDisplay message={rootError} />}
                    {rootData && <p className="text-green-300">{rootData.message}</p>}
                </GlassCard>
                <GlassCard className="p-6">
                    <h3 className="text-xl font-semibold text-white mb-4">API Health Check</h3>
                    {healthLoading && <Loader />}
                    {healthError && <ErrorDisplay message={healthError} />}
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
        execute(); // Initial fetch
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

const RealtimePredictionSection = () => {
    const { data: batchData, loading, error, execute: executeBatch } = useApi('/predict/batch', { method: 'POST' });
    const [formData, setFormData] = useState({ temperature: 25.0, motion: 0, pulse: 70.0 });
    const [latestPrediction, setLatestPrediction] = useState(null);

    const handleInputChange = (e) => {
        const { name, value, type } = e.target;
        setFormData(prev => ({ ...prev, [name]: type === 'number' ? parseFloat(value) : parseInt(value) }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLatestPrediction(null);

        const windowSize = 30;
        const now = new Date();
        const sensorReadings = new Array(windowSize);

        for (let i = 0; i < windowSize; i++) {
            const timestamp = new Date(now.getTime() - (windowSize - 1 - i) * 1000).toISOString();
            
            let currentTemp, currentPulse, currentMotion;
            if (i === windowSize - 1) {
                currentTemp = formData.temperature;
                currentPulse = formData.pulse;
                currentMotion = formData.motion;
            } else {
                const pulseVariation = 2.5 * Math.sin((i / windowSize) * Math.PI * 2);
                const tempVariation = 0.5 * Math.sin((i / windowSize) * Math.PI * 2);
                const pulseNoise = (Math.random() - 0.5) * 2;
                const tempNoise = (Math.random() - 0.5) * 0.2;

                currentTemp = formData.temperature + tempVariation + tempNoise;
                currentPulse = formData.pulse + pulseVariation + pulseNoise;
                currentMotion = Math.random() < 0.05 ? 1 : 0;
            }

            sensorReadings[i] = {
                temperature: currentTemp,
                motion: currentMotion,
                pulse: currentPulse,
                timestamp: timestamp,
                sensor_id: "realtime_test_window"
            };
        }

        const payload = { 
            data: sensorReadings,
            return_probabilities: true
        };
        
        const result = await executeBatch(payload);
        
        if (result && result.predictions && result.predictions.length > 0) {
            setLatestPrediction(result.predictions[result.predictions.length - 1]);
        }
    };
    
    const pieData = latestPrediction?.probabilities ? Object.entries(latestPrediction.probabilities).map(([key, value]) => ({
        name: ANOMALY_TYPES[key] || key,
        value,
    })) : [];

    return (
        <div>
            <h2 className="text-3xl font-bold text-white mb-6">Real-time Prediction</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <GlassCard className="p-6">
                    <h3 className="text-xl font-semibold text-white mb-4">Sensor Input</h3>
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-1">Temperature (°C)</label>
                            <input type="number" name="temperature" value={formData.temperature} onChange={handleInputChange} className="w-full bg-white/10 border border-white/20 rounded-lg p-2 text-white focus:ring-cyan-400 focus:border-cyan-400" step="0.1" />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-1">Motion</label>
                            <select name="motion" value={formData.motion} onChange={handleInputChange} className="w-full bg-white/10 border border-white/20 rounded-lg p-2 text-white focus:ring-cyan-400 focus:border-cyan-400">
                                <option value={0}>No Motion</option>
                                <option value={1}>Motion Detected</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-1">Pulse (BPM)</label>
                            <input type="number" name="pulse" value={formData.pulse} onChange={handleInputChange} className="w-full bg-white/10 border border-white/20 rounded-lg p-2 text-white focus:ring-cyan-400 focus:border-cyan-400" step="0.1" />
                        </div>
                        <button type="submit" disabled={loading} className="w-full bg-cyan-500 text-white font-bold py-2 px-4 rounded-lg hover:bg-cyan-600 transition duration-300 disabled:bg-gray-500">
                            {loading ? 'Analyzing...' : 'Predict Anomaly'}
                        </button>
                    </form>
                </GlassCard>
                <div>
                    <GlassCard className="p-6">
                        <h3 className="text-xl font-semibold text-white mb-4">Prediction Result</h3>
                        {loading && <Loader />}
                        {error && <ErrorDisplay message={error} />}
                        {latestPrediction && (
                            <div className="space-y-4">
                                <div className={`p-4 rounded-lg border-2 ${latestPrediction.prediction === 'normal' ? 'border-green-400 bg-green-500/20' : 'border-red-400 bg-red-500/20'}`}>
                                    <p className="text-lg font-bold text-white">Prediction: <span className={latestPrediction.prediction === 'normal' ? 'text-green-300' : 'text-red-300'}>{latestPrediction.prediction}</span></p>
                                    <p className="text-white">Confidence: {(latestPrediction.confidence * 100).toFixed(2)}%</p>
                                </div>
                                 {pieData.length > 0 && (
                                    <div>
                                        <h4 className="font-semibold text-white mb-2">Probabilities</h4>
                                        <ResponsiveContainer width="100%" height={200}>
                                            <PieChart>
                                                <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={60} fill="#8884d8">
                                                    {pieData.map((entry, index) => (
                                                        <Cell key={`cell-${index}`} fill={ANOMALY_COLORS[entry.name] || '#95A5A6'} />
                                                    ))}
                                                </Pie>
                                                <Tooltip />
                                                <Legend />
                                            </PieChart>
                                        </ResponsiveContainer>
                                    </div>
                                )}
                            </div>
                        )}
                    </GlassCard>
                    {latestPrediction && latestPrediction.prediction !== 'normal' && (
                        <GeminiAnalysisCard prediction={latestPrediction} sensorData={formData} />
                    )}
                </div>
            </div>
        </div>
    );
};

const BatchPredictionSection = () => {
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
            alert("Please select a file first.");
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
                        return headers.reduce((obj, header, index) => {
                            let value = values[index]?.trim();
                            obj[header] = isNaN(value) || value === '' ? value : parseFloat(value);
                            return obj;
                        }, {});
                    });
                }
                const payload = { data: jsonData };
                await execute(payload);
            } catch (err) {
                alert("Failed to parse file: " + err.message);
            }
        };
        reader.readAsText(file);
    };

    return (
        <div>
            <h2 className="text-3xl font-bold text-white mb-6">Batch Prediction</h2>
            <GlassCard className="p-6 mb-6">
                <h3 className="text-xl font-semibold text-white mb-4">Upload Data File</h3>
                <div 
                    className="border-2 border-dashed border-gray-400 rounded-lg p-8 text-center cursor-pointer hover:border-cyan-400 hover:bg-white/5 transition"
                    onDrop={handleFileDrop}
                    onDragOver={(e) => e.preventDefault()}
                    onClick={() => fileInputRef.current.click()}
                >
                    <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept=".csv,.json" />
                    <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                    {file ? (
                        <p className="text-white">{file.name}</p>
                    ) : (
                        <p className="text-gray-300">Drag & drop a CSV or JSON file here, or click to select</p>
                    )}
                </div>
                <button onClick={handleSubmit} disabled={loading || !file} className="mt-4 w-full bg-cyan-500 text-white font-bold py-2 px-4 rounded-lg hover:bg-cyan-600 transition duration-300 disabled:bg-gray-500">
                    {loading ? 'Processing...' : 'Process Batch'}
                </button>
            </GlassCard>
            
            {loading && <Loader message="Processing batch file..." />}
            {error && <ErrorDisplay message={error} />}
            {data && (
                <GlassCard className="p-6">
                    <h3 className="text-xl font-semibold text-white mb-4">Batch Results</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                        <div className="bg-white/10 p-4 rounded-lg text-center">
                            <p className="text-gray-300 text-sm">Total Predictions</p>
                            <p className="text-white text-2xl font-bold">{data.summary.total_predictions}</p>
                        </div>
                        <div className="bg-white/10 p-4 rounded-lg text-center">
                            <p className="text-gray-300 text-sm">Anomalies Found</p>
                            <p className="text-red-400 text-2xl font-bold">{Object.entries(data.summary.class_distribution).reduce((acc, [key, val]) => key !== 'normal' ? acc + val : acc, 0)}</p>
                        </div>
                        <div className="bg-white/10 p-4 rounded-lg text-center">
                            <p className="text-gray-300 text-sm">Avg. Confidence</p>
                            <p className="text-white text-2xl font-bold">{(data.summary.average_confidence * 100).toFixed(2)}%</p>
                        </div>
                    </div>
                    <div className="overflow-x-auto max-h-96">
                        <table className="w-full text-left text-sm text-white">
                            <thead className="bg-white/20 sticky top-0">
                                <tr>
                                    <th className="p-2">Timestamp</th>
                                    <th className="p-2">Prediction</th>
                                    <th className="p-2">Confidence</th>
                                </tr>
                            </thead>
                            <tbody>
                                {data.predictions.map((p, i) => (
                                    <tr key={i} className="border-b border-white/10 hover:bg-white/5">
                                        <td className="p-2">{new Date(p.timestamp).toLocaleString()}</td>
                                        <td className="p-2">
                                            <span className={`px-2 py-1 rounded-full text-xs font-semibold ${p.prediction === 'normal' ? 'bg-green-500/30 text-green-300' : 'bg-red-500/30 text-red-300'}`}>
                                                {p.prediction}
                                            </span>
                                        </td>
                                        <td className="p-2">{(p.confidence * 100).toFixed(2)}%</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </GlassCard>
            )}
        </div>
    );
};

const SimulateDataSection = () => {
    const { data, loading, error, execute } = useApi('/simulate');
    const [numSamples, setNumSamples] = useState(200);
    const [includeAnomalies, setIncludeAnomalies] = useState(true);

    const handleSimulate = () => {
        execute(null, { num_samples: numSamples, include_anomalies: includeAnomalies });
    };

    return (
        <div>
            <h2 className="text-3xl font-bold text-white mb-6">Simulate Sensor Data</h2>
            <GlassCard className="p-6 mb-6">
                <div className="flex flex-wrap items-center gap-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">Number of Samples</label>
                        <input type="number" value={numSamples} onChange={(e) => setNumSamples(parseInt(e.target.value))} className="bg-white/10 border border-white/20 rounded-lg p-2 text-white w-32" />
                    </div>
                    <div className="flex items-center pt-6">
                        <input type="checkbox" id="anomalies" checked={includeAnomalies} onChange={(e) => setIncludeAnomalies(e.target.checked)} className="h-4 w-4 rounded border-gray-300 bg-white/10 text-cyan-500 focus:ring-cyan-500" />
                        <label htmlFor="anomalies" className="ml-2 text-sm text-gray-300">Include Anomalies</label>
                    </div>
                    <button onClick={handleSimulate} disabled={loading} className="ml-auto bg-cyan-500 text-white font-bold py-2 px-4 rounded-lg hover:bg-cyan-600 transition duration-300 disabled:bg-gray-500">
                        {loading ? 'Generating...' : 'Generate Data'}
                    </button>
                </div>
            </GlassCard>

            {loading && <Loader message="Simulating data..." />}
            {error && <ErrorDisplay message={error} />}
            {data && data.data && (
                <GlassCard className="p-6">
                    <h3 className="text-xl font-semibold text-white mb-4">Simulated Data Visualization</h3>
                    <ResponsiveContainer width="100%" height={400}>
                        <AreaChart data={data.data.map(d => ({...d, timestamp: new Date(d.timestamp).toLocaleTimeString()}))}>
                            <defs>
                                <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
                                    <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
                                </linearGradient>
                                <linearGradient id="colorPulse" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#82ca9d" stopOpacity={0.8}/>
                                    <stop offset="95%" stopColor="#82ca9d" stopOpacity={0}/>
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.2)" />
                            <XAxis dataKey="timestamp" stroke="rgba(255, 255, 255, 0.7)" />
                            <YAxis yAxisId="left" stroke="rgba(255, 255, 255, 0.7)" />
                            <YAxis yAxisId="right" orientation="right" stroke="rgba(255, 255, 255, 0.7)" />
                            <Tooltip content={<CustomTooltip />} />
                            <Area yAxisId="left" type="monotone" dataKey="temperature" stroke="#8884d8" fillOpacity={1} fill="url(#colorTemp)" />
                            <Area yAxisId="right" type="monotone" dataKey="pulse" stroke="#82ca9d" fillOpacity={1} fill="url(#colorPulse)" />
                        </AreaChart>
                    </ResponsiveContainer>
                </GlassCard>
            )}
        </div>
    );
};

// --- Main App Component ---
export default function App() {
    const [activeSection, setActiveSection] = useState('dashboard');

    const renderSection = () => {
        switch (activeSection) {
            case 'dashboard': return <DashboardSection />;
            case 'status': return <SystemStatusSection />;
            case 'models': return <ModelsSection />;
            case 'predict': return <RealtimePredictionSection />;
            case 'batch': return <BatchPredictionSection />;
            case 'simulate': return <SimulateDataSection />;
            default: return <DashboardSection />;
        }
    };

    return (
        <div className="min-h-screen bg-gray-900 text-white font-sans bg-cover bg-center" style={{backgroundImage: "url('https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?q=80&w=2071&auto=format&fit=crop')"}}>
            <div className="flex h-screen p-4 gap-4">
                <aside className="w-64 flex-shrink-0">
                    <Sidebar activeSection={activeSection} setActiveSection={setActiveSection} />
                </aside>
                <main className="flex-1 overflow-y-auto pr-2">
                     <div className="h-full rounded-2xl p-6 transition-all duration-500">
                        {renderSection()}
                    </div>
                </main>
            </div>
        </div>
    );
}
