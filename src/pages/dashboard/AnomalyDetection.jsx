import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import { useDispatch } from 'react-redux';
import { setLoading, setToast } from '../../state/uiSlice';

const timeSeries = Array.from({ length: 100 }, (_, i) => ({
  x: i,
  y: Math.sin(i / 10) + (Math.random() - 0.5) * 0.2,
  anomaly: i === 30 || i === 60 || i === 85,
  score: Math.random(),
}));
const threshold = 0.8;

export default function AnomalyDetection() {
  const dispatch = useDispatch();
  const [data] = useState(timeSeries);

  const line = {
    x: data.map((d) => d.x),
    y: data.map((d) => d.y),
    type: 'scatter',
    mode: 'lines',
    name: 'Series',
    line: { color: '#0ff1ce' },
  };
  const anomalies = {
    x: data.filter((d) => d.anomaly).map((d) => d.x),
    y: data.filter((d) => d.anomaly).map((d) => d.y),
    mode: 'markers',
    type: 'scatter',
    name: 'Anomaly',
    marker: { color: 'red', size: 10, symbol: 'circle' },
  };
  const score = {
    x: data.map((d) => d.x),
    y: data.map((d) => d.score),
    type: 'scatter',
    mode: 'lines',
    name: 'Anomaly Score',
    yaxis: 'y2',
    line: { color: '#7f5af0', dash: 'dot' },
  };
  const thresholdLine = {
    x: data.map((d) => d.x),
    y: Array(data.length).fill(threshold),
    type: 'scatter',
    mode: 'lines',
    name: 'Threshold',
    yaxis: 'y2',
    line: { color: 'red', dash: 'dash' },
  };

  return (
    <div className="max-w-4xl mx-auto bg-card p-8 rounded-lg shadow-lg border border-border mt-8">
      <h2 className="text-2xl font-bold mb-6 text-center">Anomaly Detection</h2>
      <Plot
        data={[line, anomalies, score, thresholdLine]}
        layout={{
          paper_bgcolor: '#23233a',
          plot_bgcolor: '#23233a',
          font: { color: '#fff' },
          xaxis: { title: 'Time' },
          yaxis: { title: 'Value' },
          yaxis2: {
            title: 'Anomaly Score',
            overlaying: 'y',
            side: 'right',
            showgrid: false,
          },
          legend: { orientation: 'h', y: -0.2 },
          margin: { t: 30, b: 60 },
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%', height: '400px' }}
      />
    </div>
  );
}
