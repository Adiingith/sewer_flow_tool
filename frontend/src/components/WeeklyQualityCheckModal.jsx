import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { X, PlusCircle, Trash2 } from 'lucide-react';
import axios from 'axios';
import WeeklyCheckRow from './WeeklyCheckRow';

const API_BASE_URL = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

const WeeklyQualityCheckModal = ({ monitors, onClose }) => {
  const [historicalChecks, setHistoricalChecks] = useState({});
  const [newChecks, setNewChecks] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const monitorIds = useMemo(() => monitors.map(m => m.id), [monitors]);

  const fetchWeeklyChecks = useCallback(async () => {
    if (!monitorIds || monitorIds.length === 0) {
      setHistoricalChecks({});
      return;
    }
    
    setIsLoading(true);
    setError('');
    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/weekly_quality_check/by_monitors`, monitorIds);
      setHistoricalChecks(response.data || {});
    } catch (err) {
      console.error("Error fetching weekly checks:", err);
      setError('Failed to load weekly check data.');
    } finally {
      setIsLoading(false);
    }
  }, [monitorIds]);

  useEffect(() => {
    fetchWeeklyChecks();
  }, [fetchWeeklyChecks]);
  
  const handleInputChange = useCallback((monitorId, interim, field, value) => {
    setNewChecks(prev =>
      prev.map(check =>
        (check.monitor_id === monitorId && check.interim === interim) ? { ...check, [field]: value } : check
      )
    );
  }, []);

  const handleAddNewCheckForAll = useCallback(() => {
    const checksToAdd = [];
    monitors.forEach(monitor => {
      // Check if a new row for this monitor already exists to prevent duplicates
      if (newChecks.some(c => c.monitor_id === monitor.id)) {
        return;
      }

      const existingChecks = [
        ...(historicalChecks[monitor.id] || []),
        ...newChecks.filter(c => c.monitor_id === monitor.id)
      ];
      
      const maxInterim = existingChecks.reduce((max, check) => {
        const interimNumber = check.interim ? parseInt(String(check.interim).replace('interim', ''), 10) || 0 : 0;
        return Math.max(max, interimNumber);
      }, 0);

      checksToAdd.push({
        monitor_id: monitor.id,
        monitor_name: monitor.monitor_name,
        interim: `interim${maxInterim + 1}`,
        silt_mm: '',
        comments: '',
        actions: '',
        check_date: new Date().toISOString().split('T')[0], // Prefill today's date
      });
    });
    setNewChecks(prev => [...prev, ...checksToAdd]);
  }, [monitors, historicalChecks, newChecks]);

  const handleAddNewCheckForMonitor = useCallback((monitor) => {
    const existingChecks = [
      ...(historicalChecks[monitor.id] || []),
      ...newChecks.filter(c => c.monitor_id === monitor.id)
    ];
    
    const maxInterim = existingChecks.reduce((max, check) => {
      const interimNumber = check.interim ? parseInt(String(check.interim).replace('interim', ''), 10) || 0 : 0;
      return Math.max(max, interimNumber);
    }, 0);
  
    const newCheck = {
      monitor_id: monitor.id,
      monitor_name: monitor.monitor_name,
      interim: `interim${maxInterim + 1}`,
      silt_mm: '',
      comments: '',
      actions: '',
      check_date: new Date().toISOString().split('T')[0],
    };
  
    setNewChecks(prev => [...prev, newCheck]);
  }, [historicalChecks, newChecks]);

  const handleRemoveNewCheck = useCallback((monitorId, interim) => {
    setNewChecks(prev => prev.filter(check => !(check.monitor_id === monitorId && check.interim === interim)));
  }, []);

  const handleSubmit = async () => {
    if (newChecks.length === 0) {
      onClose(); // Nothing to save
      return;
    }

    setIsLoading(true);
    setError('');

    const payloads = newChecks.map(check => {
      const { monitor_name, ...payload } = check;
      return {
          ...payload,
          silt_mm: payload.silt_mm ? parseInt(payload.silt_mm, 10) : null,
      };
    });

    try {
      await axios.post(`${API_BASE_URL}/api/v1/weekly_quality_check/batch`, payloads);
      onClose();
    } catch (err) {
      console.error("Error saving weekly checks:", err);
      setError(err.response?.data?.detail || 'Failed to save checks. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };
  
  const renderTableForMonitor = (monitor) => {
    const historical = historicalChecks[monitor.id] || [];
    const newChecksForMonitor = newChecks.filter(c => c.monitor_id === monitor.id);

    return (
      <div key={monitor.id} className="mb-8">
        <div className="flex justify-between items-center bg-gray-100 p-2 rounded-t-md">
          <h3 className="text-lg font-semibold text-gray-900">{monitor.monitor_name}</h3>
          <button
            type="button"
            onClick={() => handleAddNewCheckForMonitor(monitor)}
            className="flex items-center px-3 py-1 text-xs font-medium text-white bg-green-600 rounded-md hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
            disabled={isLoading}
          >
            <PlusCircle size={16} className="mr-1" />
            Add
          </button>
        </div>
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-1/12">Interim</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[15%]">Check Date</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[10%]">Silt (mm)</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[25%]">Comments</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[25%]">Actions</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[10%]">Edit</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {historical.map(check => {
              const displayComments = (check.comments && typeof check.comments === 'object') ? check.comments.notes : check.comments;
              const displayActions = (check.actions && typeof check.actions === 'object') ? check.actions.actions : check.actions;
              return (
                <tr key={check.id}>
                  <td className="px-4 py-2 whitespace-nowrap">{check.interim}</td>
                  <td className="px-4 py-2 whitespace-nowrap">{check.check_date}</td>
                  <td className="px-4 py-2 whitespace-nowrap">{check.silt_mm}</td>
                  <td className="px-4 py-2 whitespace-normal break-words">{displayComments || ''}</td>
                  <td className="px-4 py-2 whitespace-normal break-words">{displayActions || ''}</td>
                  <td className="px-4 py-2"></td>
                </tr>
              );
            })}
            {newChecksForMonitor.map(newCheck => (
              <WeeklyCheckRow
                key={newCheck.interim}
                check={newCheck}
                handleInputChange={handleInputChange}
                handleRemoveNewCheck={handleRemoveNewCheck}
                isNew
              />
            ))}
          </tbody>
        </table>
      </div>
    );
  };
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
      <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-7xl max-h-[90vh] flex flex-col">
        <div className="flex justify-between items-center border-b pb-3">
          <h2 className="text-xl font-bold text-gray-800">Weekly Quality Check</h2>
          <div className="flex items-center space-x-4">
            <button
              type="button"
              onClick={handleAddNewCheckForAll}
              className="flex items-center px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:bg-gray-400"
              disabled={isLoading || newChecks.length >= monitors.length}
            >
              <PlusCircle size={18} className="mr-2" />
              Add New Check for All
            </button>
            <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
              <X size={24} />
            </button>
          </div>
        </div>

        {error && <div className="my-3 p-3 bg-red-100 text-red-700 rounded">{error}</div>}
        
        <div className="flex-grow overflow-y-auto mt-4 pr-2">
          {isLoading && Object.keys(historicalChecks).length === 0 ? (
            <p className="text-center py-4">Loading checks...</p>
          ) : (
            monitors.map(monitor => renderTableForMonitor(monitor))
          )}
           {monitors.length > 0 && newChecks.length === 0 && !isLoading && (
            <div className="text-center py-8 text-gray-500">
              <p>No new checks added.</p>
              <p>Click "Add New Check" to create a new weekly check for the selected monitors.</p>
            </div>
          )}
        </div>
        <div className="flex-shrink-0 flex justify-end mt-4 pt-4 border-t">
          <button 
            type="button"
            onClick={handleSubmit}
            disabled={isLoading || newChecks.length === 0} 
            className="px-6 py-2 text-sm font-medium text-white bg-teal-600 rounded-md hover:bg-teal-700 disabled:bg-gray-400"
          >
            {isLoading ? 'Saving...' : `Save ${newChecks.length} New Check(s)`}
          </button>
        </div>
      </div>
    </div>
  );
};

export default WeeklyQualityCheckModal; 