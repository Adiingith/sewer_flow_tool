import React from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

function KPICard({ title, value, change, changeType, unit = '', icon: Icon, color = 'blue' }) {
  const getTrendIcon = () => {
    if (changeType === 'up') return <TrendingUp className="w-4 h-4" />;
    if (changeType === 'down') return <TrendingDown className="w-4 h-4" />;
    return <Minus className="w-4 h-4" />;
  };

  const getTrendColor = () => {
    if (changeType === 'up') return 'text-green-600';
    if (changeType === 'down') return 'text-red-600';
    return 'text-gray-500';
  };

  const getCardColor = () => {
    const colors = {
      blue: 'border-blue-200 bg-blue-50',
      green: 'border-green-200 bg-green-50',
      orange: 'border-orange-200 bg-orange-50',
      red: 'border-red-200 bg-red-50',
    };
    return colors[color] || colors.blue;
  };

  const getIconColor = () => {
    const colors = {
      blue: 'text-blue-600',
      green: 'text-green-600',
      orange: 'text-orange-600',
      red: 'text-red-600',
    };
    return colors[color] || colors.blue;
  };

  return (
    <div className={`rounded-2xl shadow-md p-4 bg-white border ${getCardColor()}`}>
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
          <div className="flex items-baseline space-x-2">
            <p className="text-2xl font-bold text-gray-900">
              {value}
              {unit && <span className="text-lg font-normal text-gray-600 ml-1">{unit}</span>}
            </p>
            {change !== undefined && (
              <div className={`flex items-center space-x-1 ${getTrendColor()}`}>
                {getTrendIcon()}
                <span className="text-sm font-medium">{Math.abs(change)}%</span>
              </div>
            )}
          </div>
        </div>
        {Icon && (
          <div className={`flex-shrink-0 ${getIconColor()}`}>
            <Icon className="w-8 h-8" />
          </div>
        )}
      </div>
    </div>
  );
}

export default KPICard; 