import React from 'react';

export const GlassCard = ({ children, className = '', onClick }) => (
    <div 
        className={`bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 shadow-lg transition-all duration-300 hover:shadow-xl hover:bg-white/20 ${className}`}
        onClick={onClick}
    >
        {children}
    </div>
);

export const GlassButton = ({ children, className = '', onClick, disabled = false }) => (
    <button
        className={`px-4 py-2 bg-white/10 backdrop-blur-lg rounded-lg border border-white/20 
            shadow-lg transition-all duration-300 
            ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:shadow-xl hover:bg-white/20'} 
            ${className}`}
        onClick={onClick}
        disabled={disabled}
    >
        {children}
    </button>
);

export const GlassInput = ({ className = '', ...props }) => (
    <input
        className={`w-full bg-white/10 backdrop-blur-lg rounded-lg border border-white/20 
            p-2 text-white placeholder-gray-400 
            focus:ring-2 focus:ring-cyan-400 focus:border-transparent 
            transition-all duration-300 
            ${className}`}
        {...props}
    />
);

export const GlassSelect = ({ children, className = '', ...props }) => (
    <select
        className={`w-full bg-white/10 backdrop-blur-lg rounded-lg border border-white/20 
            p-2 text-white 
            focus:ring-2 focus:ring-cyan-400 focus:border-transparent 
            transition-all duration-300 
            ${className}`}
        {...props}
    >
        {children}
    </select>
);