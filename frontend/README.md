# WSN Anomaly Detection Frontend

This is the frontend application for the WSN (Wireless Sensor Network) Anomaly Detection System. It provides a modern, responsive interface for monitoring sensor data and detecting anomalies in real-time.

## Features

- Real-time sensor data monitoring
- Anomaly detection visualization
- System status monitoring
- Batch prediction capabilities
- Interactive data simulation
- Beautiful glass-morphism UI design

## Prerequisites

- Node.js (v14.0.0 or higher)
- npm (v6.0.0 or higher)
- Backend API running on http://127.0.0.1:8000

## Installation

1. Install dependencies:
   ```bash
   npm install
   ```

2. Create a production build:
   ```bash
   npm run build
   ```

3. Start the development server:
   ```bash
   npm start
   ```

The application will be available at http://localhost:3000

## Development

- `npm start` - Runs the app in development mode
- `npm test` - Launches the test runner
- `npm run build` - Builds the app for production
- `npm run eject` - Ejects from create-react-app

## Project Structure

```
├── public/             # Static files
├── src/                # Source files
│   ├── components/     # React components
│   ├── hooks/         # Custom React hooks
│   ├── utils/         # Utility functions
│   └── index.js       # Entry point
├── App.js             # Main application component
└── package.json       # Project dependencies
```

## Configuration

The backend API URL can be configured in `App.js`:

```javascript
const API_BASE_URL = "http://127.0.0.1:8000";
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details