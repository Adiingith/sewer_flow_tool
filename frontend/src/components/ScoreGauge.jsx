import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

function ScoreGauge({ score, size = 120, showLabel = true }) {
  const value = score || 0;
  const percentage = Math.round(value * 100);
  
  // Calculate gauge data
  const data = [
    { name: 'score', value: value * 100 },
    { name: 'remaining', value: (1 - value) * 100 }
  ];
  
  // Determine color based on score
  const getColor = (score) => {
    if (score >= 0.8) return '#10B981'; // green
    if (score >= 0.6) return '#F59E0B'; // orange  
    return '#EF4444'; // red
  };
  
  const getColorClass = (score) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-orange-600';
    return 'text-red-600';
  };
  
  const getStatusText = (score) => {
    if (score >= 0.8) return 'Excellent';
    if (score >= 0.6) return 'Good';
    return 'Needs Attention';
  };
  
  const colors = [getColor(value), '#E5E7EB'];
  
  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: size, height: size }}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              startAngle={90}
              endAngle={450}
              innerRadius={size * 0.3}
              outerRadius={size * 0.45}
              paddingAngle={0}
              dataKey="value"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={colors[index]} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        
        {/* Center score display */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div className={`text-2xl font-bold ${getColorClass(value)}`}>
            {percentage}%
          </div>
          {showLabel && (
            <div className="text-xs text-gray-500 mt-1">
              {getStatusText(value)}
            </div>
          )}
        </div>
      </div>
      
      {/* Score range indicators */}
      {showLabel && (
        <div className="mt-4 flex items-center space-x-4 text-xs">
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <span className="text-gray-600">0-59%</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
            <span className="text-gray-600">60-79%</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="text-gray-600">80-100%</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default ScoreGauge; 