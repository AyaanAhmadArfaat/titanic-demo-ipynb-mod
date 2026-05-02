require('dotenv').config();
const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const axios = require('axios');
const morgan = require('morgan');
const helmet = require('helmet');

/**
 * Titanic Survival Prediction API Gateway
 * Proxying traffic to internal FastAPI model server.
 * MLflow integration is handled by the upstream FastAPI service.
 */

const app = express();

// Configuration from Environment
const PORT = process.env.GATEWAY_PORT || 3000;
const FASTAPI_TARGET = process.env.FASTAPI_SERVICE_URL || 'http://localhost:8000';
const MLFLOW_TRACKING_URI = process.env.MLFLOW_TRACKING_URI;

// Security & Monitoring Middleware
app.use(helmet());
app.use(morgan('combined'));
app.use(express.json());

// Health Check
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'gateway_up', mlflow: !!MLFLOW_TRACKING_URI });
});

// Proxy Configuration
const apiProxy = createProxyMiddleware({
  target: FASTAPI_TARGET,
  changeOrigin: true,
  pathRewrite: { '^/api': '' },
  onProxyReq: (proxyReq, req) => {
    // Inject headers for tracking/logging if necessary
    proxyReq.setHeader('X-Gateway-Trace-Id', Date.now().toString());
  },
  onError: (err, req, res) => {
    console.error('Proxy Error:', err);
    res.status(502).json({ error: 'Backend unreachable' });
  }
});

// Route Proxying
// Routes to FastAPI: /api/predict, /api/models, etc.
app.use('/api', apiProxy);

// Global Error Handler
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Internal Gateway Error' });
});

// Start Server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 API Gateway listening on port ${PORT}`);
  console.log(`🔗 Proxying to FastAPI service at ${FASTAPI_TARGET}`);
  if (MLFLOW_TRACKING_URI) {
    console.log(`📊 MLflow Tracking Enabled: ${MLFLOW_TRACKING_URI}`);
  }
});