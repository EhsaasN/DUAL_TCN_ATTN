import React from 'react';

const metrics = [
  { label: 'Precision', value: '0.93', color: 'bg-primary' },
  { label: 'Recall', value: '0.89', color: 'bg-accent' },
  { label: 'F1-score', value: '0.91', color: 'bg-primary' },
  { label: 'AUC', value: '0.96', color: 'bg-accent' },
];

const pastRuns = [
  { dataset: 'Sensor A', date: '2025-09-29', score: '0.91' },
  { dataset: 'Sensor B', date: '2025-09-28', score: '0.89' },
  { dataset: 'Sensor C', date: '2025-09-27', score: '0.92' },
];

export default function Results() {
  return (
    <div className="max-w-4xl mx-auto bg-card p-8 rounded-lg shadow-lg border border-border mt-8">
      <h2 className="text-2xl font-bold mb-6 text-center">Results</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {metrics.map((m) => (
          <div key={m.label} className={`rounded-lg p-4 text-center shadow-md ${m.color}`}>
            <div className="text-lg font-bold">{m.label}</div>
            <div className="text-2xl font-extrabold mt-2">{m.value}</div>
          </div>
        ))}
      </div>
      <h3 className="text-xl font-semibold mb-4">Past Runs</h3>
      <table className="w-full text-left rounded-lg overflow-hidden">
        <thead>
          <tr className="bg-surface text-primary">
            <th className="py-2 px-4">Dataset</th>
            <th className="py-2 px-4">Date</th>
            <th className="py-2 px-4">F1-score</th>
          </tr>
        </thead>
        <tbody>
          {pastRuns.map((run, i) => (
            <tr key={i} className="border-b border-border hover:bg-accent/10">
              <td className="py-2 px-4">{run.dataset}</td>
              <td className="py-2 px-4">{run.date}</td>
              <td className="py-2 px-4">{run.score}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
