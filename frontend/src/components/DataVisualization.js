import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, BarChart, Bar } from 'recharts';
import { GlassCard } from './GlassCard';

const DataVisualization = ({ data, type = 'area', height = 300 }) => {
    if (!data || data.length === 0) {
        return (
            <GlassCard className="p-4 text-center text-gray-400">
                No data available for visualization
            </GlassCard>
        );
    }

    const renderChart = () => {
        const commonProps = {
            data,
            margin: { top: 10, right: 30, left: 0, bottom: 0 },
        };

        switch (type) {
            case 'area':
                return (
                    <AreaChart {...commonProps}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="timestamp" stroke="#fff" />
                        <YAxis stroke="#fff" />
                        <Tooltip content={CustomTooltip} />
                        <Area
                            type="monotone"
                            dataKey="temperature"
                            stroke="#8884d8"
                            fill="#8884d8"
                            fillOpacity={0.3}
                        />
                        <Area
                            type="monotone"
                            dataKey="pulse"
                            stroke="#82ca9d"
                            fill="#82ca9d"
                            fillOpacity={0.3}
                        />
                    </AreaChart>
                );

            case 'line':
                return (
                    <LineChart {...commonProps}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="timestamp" stroke="#fff" />
                        <YAxis stroke="#fff" />
                        <Tooltip content={CustomTooltip} />
                        <Line
                            type="monotone"
                            dataKey="temperature"
                            stroke="#8884d8"
                            strokeWidth={2}
                        />
                        <Line
                            type="monotone"
                            dataKey="pulse"
                            stroke="#82ca9d"
                            strokeWidth={2}
                        />
                    </LineChart>
                );

            case 'bar':
                return (
                    <BarChart {...commonProps}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="timestamp" stroke="#fff" />
                        <YAxis stroke="#fff" />
                        <Tooltip content={CustomTooltip} />
                        <Bar dataKey="temperature" fill="#8884d8" fillOpacity={0.8} />
                        <Bar dataKey="pulse" fill="#82ca9d" fillOpacity={0.8} />
                    </BarChart>
                );

            default:
                return null;
        }
    };

    return (
        <GlassCard className="p-4">
            <ResponsiveContainer width="100%" height={height}>
                {renderChart()}
            </ResponsiveContainer>
        </GlassCard>
    );
};

const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-gray-800/90 backdrop-blur-lg p-3 rounded-lg border border-gray-700 shadow-lg">
                <p className="text-white font-medium mb-1">{label}</p>
                {payload.map((entry, index) => (
                    <p
                        key={index}
                        style={{ color: entry.color }}
                        className="text-sm"
                    >
                        {`${entry.name}: ${entry.value}`}
                    </p>
                ))}
            </div>
        );
    }
    return null;
};

export default DataVisualization;