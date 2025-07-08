import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';

const EarlyExitAnalysis = () => {
  const [activeTab, setActiveTab] = useState('set1');

  // Parse Set 1 data (single exit configurations)
  const set1Data = [
    { block: 0, submodelParams: 183908, overallAccuracy: 80.61, exit1Accuracy: 78.49, exit1Samples: 0.93, totalParams: 388808 },
    { block: 1, submodelParams: 183908, overallAccuracy: 80.52, exit1Accuracy: 83.33, exit1Samples: 2.16, totalParams: 388808 },
    { block: 2, submodelParams: 183908, overallAccuracy: 80.96, exit1Accuracy: 87.90, exit1Samples: 3.14, totalParams: 388808 },
    { block: 3, submodelParams: 314980, overallAccuracy: 80.73, exit1Accuracy: 92.95, exit1Samples: 4.54, totalParams: 519880 },
    { block: 4, submodelParams: 314980, overallAccuracy: 81.22, exit1Accuracy: 92.09, exit1Samples: 8.72, totalParams: 519880 },
    { block: 5, submodelParams: 314980, overallAccuracy: 80.48, exit1Accuracy: 92.64, exit1Samples: 14.14, totalParams: 519880 },
    { block: 6, submodelParams: 314980, overallAccuracy: 80.81, exit1Accuracy: 93.54, exit1Samples: 17.65, totalParams: 519880 },
    { block: 7, submodelParams: 577124, overallAccuracy: 80.30, exit1Accuracy: 92.46, exit1Samples: 29.06, totalParams: 782024 },
    { block: 8, submodelParams: 577124, overallAccuracy: 80.14, exit1Accuracy: 93.72, exit1Samples: 35.01, totalParams: 782024 },
    { block: 9, submodelParams: 577124, overallAccuracy: 79.83, exit1Accuracy: 91.49, exit1Samples: 46.08, totalParams: 782024 },
    { block: 10, submodelParams: 577124, overallAccuracy: 80.75, exit1Accuracy: 92.63, exit1Samples: 52.53, totalParams: 782024 },
    { block: 11, submodelParams: 577124, overallAccuracy: 80.81, exit1Accuracy: 91.44, exit1Samples: 60.87, totalParams: 782024 },
    { block: 12, submodelParams: 577124, overallAccuracy: 80.45, exit1Accuracy: 90.48, exit1Samples: 67.52, totalParams: 782024 },
    { block: 13, submodelParams: 1101412, overallAccuracy: 81.28, exit1Accuracy: 89.20, exit1Samples: 80.11, totalParams: 1306312 },
    { block: 14, submodelParams: 1101412, overallAccuracy: 81.31, exit1Accuracy: 87.75, exit1Samples: 86.86, totalParams: 1306312 },
    { block: 15, submodelParams: 1101412, overallAccuracy: 81.60, exit1Accuracy: 87.64, exit1Samples: 88.10, totalParams: 1306312 }
  ];

  // Parse Set 2 data (dual exit configurations)
  const set2Data = {
    exit1: [
      { secondExit: 4, exit1Params: 183908, exit1Accuracy: 77.12, exit1Samples: 1.18, overallAccuracy: 79.65 },
      { secondExit: 5, exit1Params: 183908, exit1Accuracy: 82.26, exit1Samples: 0.62, overallAccuracy: 79.43 },
      { secondExit: 6, exit1Params: 183908, exit1Accuracy: 75.61, exit1Samples: 1.23, overallAccuracy: 79.21 },
      { secondExit: 7, exit1Params: 183908, exit1Accuracy: 68.97, exit1Samples: 0.58, overallAccuracy: 78.79 },
      { secondExit: 8, exit1Params: 183908, exit1Accuracy: 81.54, exit1Samples: 0.65, overallAccuracy: 78.40 },
      { secondExit: 9, exit1Params: 183908, exit1Accuracy: 80.00, exit1Samples: 0.45, overallAccuracy: 78.27 },
      { secondExit: 10, exit1Params: 183908, exit1Accuracy: 70.79, exit1Samples: 0.89, overallAccuracy: 78.66 },
      { secondExit: 11, exit1Params: 183908, exit1Accuracy: 77.78, exit1Samples: 0.72, overallAccuracy: 78.49 },
      { secondExit: 12, exit1Params: 183908, exit1Accuracy: 76.09, exit1Samples: 0.92, overallAccuracy: 78.78 },
      { secondExit: 13, exit1Params: 183908, exit1Accuracy: 68.83, exit1Samples: 0.77, overallAccuracy: 79.63 },
      { secondExit: 14, exit1Params: 183908, exit1Accuracy: 76.00, exit1Samples: 0.75, overallAccuracy: 80.53 }
    ],
    exit2: [
      { secondExit: 4, exit2Params: 314980, exit2Accuracy: 84.54, exit2Samples: 10.67, overallAccuracy: 79.65 },
      { secondExit: 5, exit2Params: 314980, exit2Accuracy: 85.82, exit2Samples: 18.48, overallAccuracy: 79.43 },
      { secondExit: 6, exit2Params: 314980, exit2Accuracy: 85.63, exit2Samples: 23.03, overallAccuracy: 79.21 },
      { secondExit: 7, exit2Params: 577124, exit2Accuracy: 87.34, exit2Samples: 31.98, overallAccuracy: 78.79 },
      { secondExit: 8, exit2Params: 577124, exit2Accuracy: 86.54, exit2Samples: 42.73, overallAccuracy: 78.40 },
      { secondExit: 9, exit2Params: 577124, exit2Accuracy: 87.07, exit2Samples: 52.92, overallAccuracy: 78.27 },
      { secondExit: 10, exit2Params: 577124, exit2Accuracy: 87.41, exit2Samples: 58.62, overallAccuracy: 78.66 },
      { secondExit: 11, exit2Params: 577124, exit2Accuracy: 86.24, exit2Samples: 68.22, overallAccuracy: 78.49 },
      { secondExit: 12, exit2Params: 577124, exit2Accuracy: 85.66, exit2Samples: 75.19, overallAccuracy: 78.78 },
      { secondExit: 13, exit2Params: 1101412, exit2Accuracy: 84.78, exit2Samples: 87.24, overallAccuracy: 79.63 },
      { secondExit: 14, exit2Params: 1101412, exit2Accuracy: 84.11, exit2Samples: 92.12, overallAccuracy: 80.53 }
    ]
  };

  const set2FirstExitData = [
    { firstExit: 1, secondExit: 4, exit1Params: 183908, exit1Accuracy: 93.52, exit1Samples: 1.08, overallAccuracy: 79.48 },
    { firstExit: 1, secondExit: 5, exit1Params: 183908, exit1Accuracy: 82.08, exit1Samples: 1.73, overallAccuracy: 79.47 },
    { firstExit: 1, secondExit: 6, exit1Params: 183908, exit1Accuracy: 81.68, exit1Samples: 1.31, overallAccuracy: 79.74 },
    { firstExit: 1, secondExit: 7, exit1Params: 183908, exit1Accuracy: 83.72, exit1Samples: 0.86, overallAccuracy: 78.80 },
    { firstExit: 1, secondExit: 8, exit1Params: 183908, exit1Accuracy: 88.04, exit1Samples: 0.92, overallAccuracy: 78.42 },
    { firstExit: 1, secondExit: 9, exit1Params: 183908, exit1Accuracy: 87.10, exit1Samples: 1.24, overallAccuracy: 78.61 },
    { firstExit: 1, secondExit: 10, exit1Params: 183908, exit1Accuracy: 88.89, exit1Samples: 0.99, overallAccuracy: 78.69 },
    { firstExit: 1, secondExit: 11, exit1Params: 183908, exit1Accuracy: 83.96, exit1Samples: 1.06, overallAccuracy: 77.92 },
    { firstExit: 1, secondExit: 12, exit1Params: 183908, exit1Accuracy: 86.31, exit1Samples: 1.68, overallAccuracy: 78.77 },
    { firstExit: 1, secondExit: 13, exit1Params: 183908, exit1Accuracy: 83.76, exit1Samples: 1.97, overallAccuracy: 80.23 },
    { firstExit: 1, secondExit: 14, exit1Params: 183908, exit1Accuracy: 86.02, exit1Samples: 1.86, overallAccuracy: 80.88 },
    
    { firstExit: 2, secondExit: 4, exit1Params: 183908, exit1Accuracy: 86.79, exit1Samples: 2.12, overallAccuracy: 79.74 },
    { firstExit: 2, secondExit: 5, exit1Params: 183908, exit1Accuracy: 91.62, exit1Samples: 1.79, overallAccuracy: 79.57 },
    { firstExit: 2, secondExit: 6, exit1Params: 183908, exit1Accuracy: 83.91, exit1Samples: 2.61, overallAccuracy: 79.36 },
    { firstExit: 2, secondExit: 7, exit1Params: 183908, exit1Accuracy: 88.95, exit1Samples: 1.90, overallAccuracy: 78.95 },
    { firstExit: 2, secondExit: 8, exit1Params: 183908, exit1Accuracy: 91.28, exit1Samples: 1.95, overallAccuracy: 78.39 },
    { firstExit: 2, secondExit: 9, exit1Params: 183908, exit1Accuracy: 81.06, exit1Samples: 2.27, overallAccuracy: 78.87 },
    { firstExit: 2, secondExit: 10, exit1Params: 183908, exit1Accuracy: 83.95, exit1Samples: 1.62, overallAccuracy: 78.60 },
    { firstExit: 2, secondExit: 11, exit1Params: 183908, exit1Accuracy: 87.10, exit1Samples: 2.48, overallAccuracy: 78.18 },
    { firstExit: 2, secondExit: 12, exit1Params: 183908, exit1Accuracy: 83.67, exit1Samples: 2.94, overallAccuracy: 78.01 },
    { firstExit: 2, secondExit: 13, exit1Params: 183908, exit1Accuracy: 82.22, exit1Samples: 2.25, overallAccuracy: 79.34 },
    { firstExit: 2, secondExit: 14, exit1Params: 183908, exit1Accuracy: 87.56, exit1Samples: 2.01, overallAccuracy: 80.83 },

    { firstExit: 3, secondExit: 4, exit1Params: 314980, exit1Accuracy: 91.56, exit1Samples: 3.20, overallAccuracy: 79.79 },
    { firstExit: 3, secondExit: 5, exit1Params: 314980, exit1Accuracy: 91.96, exit1Samples: 3.11, overallAccuracy: 79.26 },
    { firstExit: 3, secondExit: 6, exit1Params: 314980, exit1Accuracy: 90.88, exit1Samples: 3.51, overallAccuracy: 79.37 },
    { firstExit: 3, secondExit: 7, exit1Params: 314980, exit1Accuracy: 90.65, exit1Samples: 3.21, overallAccuracy: 78.97 },
    { firstExit: 3, secondExit: 8, exit1Params: 314980, exit1Accuracy: 91.64, exit1Samples: 2.99, overallAccuracy: 78.54 },
    { firstExit: 3, secondExit: 9, exit1Params: 314980, exit1Accuracy: 92.48, exit1Samples: 3.59, overallAccuracy: 78.47 },
    { firstExit: 3, secondExit: 10, exit1Params: 314980, exit1Accuracy: 87.71, exit1Samples: 4.07, overallAccuracy: 77.85 },
    { firstExit: 3, secondExit: 11, exit1Params: 314980, exit1Accuracy: 91.71, exit1Samples: 3.74, overallAccuracy: 78.14 },
    { firstExit: 3, secondExit: 12, exit1Params: 314980, exit1Accuracy: 90.25, exit1Samples: 3.18, overallAccuracy: 78.26 },
    { firstExit: 3, secondExit: 13, exit1Params: 314980, exit1Accuracy: 89.91, exit1Samples: 4.66, overallAccuracy: 79.52 },
    { firstExit: 3, secondExit: 14, exit1Params: 314980, exit1Accuracy: 90.94, exit1Samples: 3.31, overallAccuracy: 80.65 }
  ];

  const formatNumber = (value) => {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`;
    } else if (value >= 1000) {
      return `${(value / 1000).toFixed(0)}K`;
    }
    return value.toString();
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
          <p className="font-semibold">{`Block: ${label}`}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {`${entry.dataKey}: ${entry.value}${entry.dataKey.includes('Accuracy') ? '%' : entry.dataKey.includes('Params') ? ` (${formatNumber(entry.value)})` : entry.dataKey.includes('Samples') ? '%' : ''}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full p-6 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-6 text-center text-gray-800">Early Exit Model Analysis</h1>
      
      {/* Tab Navigation */}
      <div className="flex mb-6 border-b border-gray-300">
        <button
          className={`px-6 py-3 font-medium ${activeTab === 'set1' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-600 hover:text-gray-800'}`}
          onClick={() => setActiveTab('set1')}
        >
          Set 1: Single Exit Analysis
        </button>
        <button
          className={`px-6 py-3 font-medium ${activeTab === 'set2' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-600 hover:text-gray-800'}`}
          onClick={() => setActiveTab('set2')}
        >
          Set 2: Dual Exit Analysis
        </button>
      </div>

      {activeTab === 'set1' && (
        <div className="space-y-8">
          <div className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Set 1: Overall Accuracy vs Submodel Parameters</h2>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={set1Data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="submodelParams" 
                  tickFormatter={formatNumber}
                  label={{ value: 'Submodel Parameters', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  domain={[79, 82]} 
                  label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="overallAccuracy" 
                  stroke="#2563eb" 
                  strokeWidth={3}
                  dot={{ fill: "#2563eb", strokeWidth: 2, r: 6 }}
                  name="Overall Accuracy"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Set 1: Exit Accuracy vs Sample Percentage</h2>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart data={set1Data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="exit1Samples" 
                  label={{ value: 'Samples Exited (%)', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  domain={[75, 95]} 
                  label={{ value: 'Exit 1 Accuracy (%)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Scatter 
                  dataKey="exit1Accuracy" 
                  fill="#dc2626"
                  name="Exit 1 Accuracy"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {activeTab === 'set2' && (
        <div className="space-y-8">
          <div className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Set 2: First Exit (Block 0) Performance</h2>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={set2Data.exit1}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="secondExit" 
                  label={{ value: 'Second Exit Block Position', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  yAxisId="accuracy"
                  domain={[65, 85]} 
                  label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }}
                />
                <YAxis 
                  yAxisId="samples"
                  orientation="right"
                  domain={[0, 2]}
                  label={{ value: 'Samples Exited (%)', angle: 90, position: 'insideRight' }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Line 
                  yAxisId="accuracy"
                  type="monotone" 
                  dataKey="exit1Accuracy" 
                  stroke="#dc2626" 
                  strokeWidth={3}
                  dot={{ fill: "#dc2626", strokeWidth: 2, r: 6 }}
                  name="Exit 1 Accuracy"
                />
                <Line 
                  yAxisId="samples"
                  type="monotone" 
                  dataKey="exit1Samples" 
                  stroke="#16a34a" 
                  strokeWidth={3}
                  dot={{ fill: "#16a34a", strokeWidth: 2, r: 6 }}
                  name="Exit 1 Samples %"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Set 2: Second Exit Performance vs Parameters</h2>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart data={set2Data.exit2}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="exit2Params" 
                  tickFormatter={formatNumber}
                  label={{ value: 'Exit 2 Parameters', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  domain={[83, 88]} 
                  label={{ value: 'Exit 2 Accuracy (%)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Scatter 
                  dataKey="exit2Accuracy" 
                  fill="#7c3aed"
                  name="Exit 2 Accuracy"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Set 2: First Exit Performance by Position</h2>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart data={set2FirstExitData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="firstExit" 
                  domain={[0, 4]}
                  label={{ value: 'First Exit Block Position', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  domain={[80, 95]} 
                  label={{ value: 'First Exit Accuracy (%)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Scatter 
                  dataKey="exit1Accuracy" 
                  fill="#ea580c"
                  name="First Exit Accuracy"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-lg">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">Set 2: Overall Accuracy by Configuration</h2>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={set2FirstExitData.filter(d => d.firstExit <= 3)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="secondExit" 
                  label={{ value: 'Second Exit Block Position', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  domain={[77, 81]} 
                  label={{ value: 'Overall Accuracy (%)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="overallAccuracy" 
                  stroke="#0891b2" 
                  strokeWidth={3}
                  dot={{ fill: "#0891b2", strokeWidth: 2, r: 6 }}
                  name="Overall Accuracy"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="mt-8 bg-white p-6 rounded-lg shadow-lg">
        <h3 className="text-lg font-semibold mb-4 text-gray-800">Key Insights</h3>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-blue-600 mb-2">Set 1 (Single Exit)</h4>
            <ul className="text-sm text-gray-700 space-y-1">
              <li>• Peak overall accuracy: 81.60% at block 15</li>
              <li>• Best efficiency around blocks 13-15</li>
              <li>• Exit accuracy decreases as more samples exit early</li>
              <li>• Parameter efficiency varies significantly by block position</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-purple-600 mb-2">Set 2 (Dual Exit)</h4>
            <ul className="text-sm text-gray-700 space-y-1">
              <li>• First exit typically processes 1-4% of samples</li>
              <li>• Higher parameter counts in second exit don't always improve accuracy</li>
              <li>• Best overall accuracy: 80.88% with exits at blocks 1,14</li>
              <li>• Trade-off between early exit accuracy and sample distribution</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EarlyExitAnalysis;