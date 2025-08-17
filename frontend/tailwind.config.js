/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html",
    "./App.js"
  ],
  theme: {
    extend: {
      colors: {
        'anomaly': {
          'normal': '#2ECC71',
          'dos': '#E74C3C',
          'jamming': '#F39C12',
          'tampering': '#9B59B6',
          'hardware': '#3498DB',
          'noise': '#1ABC9C',
          'unknown': '#95A5A6'
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      backdropBlur: {
        'xs': '2px',
      }
    },
  },
  plugins: [],
}