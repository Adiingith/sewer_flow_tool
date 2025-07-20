import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { X, PlusCircle, Trash2, Edit2, Save, XCircle } from 'lucide-react';
import axios from 'axios';
import WeeklyCheckRow from './WeeklyCheckRow';

const API_BASE_URL = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

// Safe render utility to avoid rendering objects as React children
const safeRender = (val) => {
  if (val == null || val === undefined) return '';
  if (typeof val === 'string' || typeof val === 'number') return val;
  if (typeof val === 'object') {
    if ('notes' in val && val.notes !== undefined) return val.notes || '';
    if ('actions' in val && val.actions !== undefined) return val.actions || '';
    try {
      return JSON.stringify(val);
    } catch {
      return '[Invalid data]';
    }
  }
  return String(val);
};

// Utility to wrap string as object for JSON fields
const wrapIfString = (val, key) => {
  if (val == null) return null;
  if (typeof val === 'string') {
    if (val.trim() === '') return null;
    if (key === 'comments') return { notes: val };
    if (key === 'actions') return { actions: val };
  }
  return val;
};

// Enhanced error handling for API calls
const getErrorMessage = (err) => {
  if (err?.response?.data) {
    const data = err.response.data;
    if (typeof data === 'object') {
      if (data.detail) return data.detail;
      if (data.msg) return data.msg;
      return JSON.stringify(data);
    }
    return String(data);
  }
  return err?.message || 'Unknown error';
};

const WeeklyQualityCheckModal = ({ monitors, onClose }) => {
  const [historicalChecks, setHistoricalChecks] = useState({});
  const [newChecks, setNewChecks] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [editingCheckId, setEditingCheckId] = useState(null);
  const [editingCheckData, setEditingCheckData] = useState({});

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
      setError(getErrorMessage(err));
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
        const interimStr = String(check.interim || '').toLowerCase();
        const interimNumber = interimStr.startsWith('interim') ? parseInt(interimStr.replace('interim', ''), 10) || 0 : 0;
        return Math.max(max, interimNumber);
      }, 0);

      checksToAdd.push({
        monitor_id: monitor.id,
        monitor_name: monitor.monitor_name,
        interim: `Interim${maxInterim + 1}`,
        silt_mm: '',
        comments: '',
        actions: '',
        data_quality_check: '',
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
      const interimStr = String(check.interim || '').toLowerCase();
      const interimNumber = interimStr.startsWith('interim') ? parseInt(interimStr.replace('interim', ''), 10) || 0 : 0;
      return Math.max(max, interimNumber);
    }, 0);
  
    const newCheck = {
      monitor_id: monitor.id,
      monitor_name: monitor.monitor_name,
      interim: `Interim${maxInterim + 1}`,
      silt_mm: '',
      comments: '',
      actions: '',
      data_quality_check: '',
      check_date: new Date().toISOString().split('T')[0],
    };
  
    setNewChecks(prev => [...prev, newCheck]);
  }, [historicalChecks, newChecks]);

  const handleRemoveNewCheck = useCallback((monitorId, interim) => {
    setNewChecks(prev => prev.filter(check => !(check.monitor_id === monitorId && check.interim === interim)));
  }, []);

  // Handle edit button click
  const handleEditClick = (check) => {
    setEditingCheckId(check.id);
    setEditingCheckData({
      check_date: check.check_date || '',
      silt_mm: check.silt_mm || '',
      comments: (check.comments && typeof check.comments === 'object' && check.comments.notes !== undefined) ? check.comments.notes : (check.comments || ''),
      actions: (check.actions && typeof check.actions === 'object' && check.actions.actions !== undefined) ? check.actions.actions : (check.actions || ''),
      interim: check.interim || '',
      data_quality_check: check.data_quality_check || '',
      device_status: check.device_status || '',
      device_status_reason: check.device_status_reason || '',
    });
  };

  // Handle input change in edit mode
  const handleEditChange = (field, value) => {
    setEditingCheckData(prev => ({ ...prev, [field]: value }));
  };

  // Handle save edit
  const handleEditSave = async (check) => {
    setIsLoading(true);
    setError('');
    try {
      const payload = {
        ...editingCheckData,
        check_date: editingCheckData.check_date ? editingCheckData.check_date : null,
        silt_mm: editingCheckData.silt_mm ? parseInt(editingCheckData.silt_mm, 10) : null,
        comments: wrapIfString(editingCheckData.comments, 'comments'),
        actions: wrapIfString(editingCheckData.actions, 'actions'),
        interim: editingCheckData.interim,
        data_quality_check: editingCheckData.data_quality_check,
        device_status: editingCheckData.device_status,
        device_status_reason: editingCheckData.device_status_reason,
      };
      await axios.put(`${API_BASE_URL}/api/v1/weekly_quality_check/${check.id}`, payload);
      setEditingCheckId(null);
      setEditingCheckData({});
      await fetchWeeklyChecks(); // Refresh data
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setIsLoading(false);
    }
  };

  // Handle cancel edit
  const handleEditCancel = () => {
    setEditingCheckId(null);
    setEditingCheckData({});
  };

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
          check_date: payload.check_date ? payload.check_date : null,
          silt_mm: payload.silt_mm ? parseInt(payload.silt_mm, 10) : null,
          comments: wrapIfString(payload.comments, 'comments'),
          actions: wrapIfString(payload.actions, 'actions'),
      };
    });

    try {
      await axios.post(`${API_BASE_URL}/api/v1/weekly_quality_check/batch`, payloads);
      onClose();
    } catch (err) {
      setError(getErrorMessage(err));
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
          <h3 className="text-lg font-semibold text-gray-900">{safeRender(monitor.monitor_name)}</h3>
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
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[12%]">Check Date</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[8%]">Silt (mm)</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[20%]">Comments</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[20%]">Actions</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[20%]">Data Quality Check</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[10%]">Edit</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {historical.map(check => {
              const isEditing = editingCheckId === check.id;
              const displayComments = (check.comments && typeof check.comments === 'object' && check.comments.notes !== undefined) ? check.comments.notes : (check.comments || '');
              const displayActions = (check.actions && typeof check.actions === 'object' && check.actions.actions !== undefined) ? check.actions.actions : (check.actions || '');
              return isEditing ? (
                <tr key={check.id} className="bg-yellow-50">
                  <td className="px-4 py-2 whitespace-nowrap">{safeRender(check.interim)}</td>
                  <td className="px-4 py-1 align-top">
                    <input
                      type="date"
                      value={editingCheckData.check_date || ''}
                      onChange={e => handleEditChange('check_date', e.target.value)}
                      className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                    />
                  </td>
                  <td className="px-4 py-1 align-top">
                    <input
                      type="number"
                      value={editingCheckData.silt_mm || ''}
                      onChange={e => handleEditChange('silt_mm', e.target.value)}
                      className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                      placeholder="mm"
                    />
                  </td>
                  <td className="px-4 py-1 align-top">
                    <textarea
                      value={editingCheckData.comments || ''}
                      onChange={e => handleEditChange('comments', e.target.value)}
                      rows="2"
                      className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                      placeholder="Enter notes..."
                    />
                  </td>
                  <td className="px-4 py-1 align-top">
                    <textarea
                      value={editingCheckData.actions || ''}
                      onChange={e => handleEditChange('actions', e.target.value)}
                      rows="2"
                      className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                      placeholder="Enter actions..."
                    />
                  </td>
                  <td className="px-4 py-1 align-top">
                    <textarea
                      value={editingCheckData.data_quality_check || ''}
                      onChange={e => handleEditChange('data_quality_check', e.target.value)}
                      rows="2"
                      className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                      placeholder="Enter data quality check..."
                    />
                  </td>
                  <td className="px-4 py-1 align-middle text-center flex flex-col gap-2">
                    <button
                      type="button"
                      onClick={() => handleEditSave(check)}
                      className="flex items-center justify-center px-2 py-1 text-xs font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 mb-1"
                      disabled={isLoading}
                    >
                      <Save size={16} className="mr-1" /> Save
                    </button>
                    <button
                      type="button"
                      onClick={handleEditCancel}
                      className="flex items-center justify-center px-2 py-1 text-xs font-medium text-gray-700 bg-gray-200 rounded-md hover:bg-gray-300"
                      disabled={isLoading}
                    >
                      <XCircle size={16} className="mr-1" /> Cancel
                    </button>
                  </td>
                </tr>
              ) : (
                <tr key={check.id}>
                  <td className="px-4 py-2 whitespace-nowrap">{safeRender(check.interim)}</td>
                  <td className="px-4 py-2 whitespace-nowrap">{safeRender(check.check_date)}</td>
                  <td className="px-4 py-2 whitespace-nowrap">{safeRender(check.silt_mm)}</td>
                  <td className="px-4 py-2 whitespace-normal break-words">{safeRender(displayComments)}</td>
                  <td className="px-4 py-2 whitespace-normal break-words">{safeRender(displayActions)}</td>
                  <td className="px-4 py-2 whitespace-normal break-words">{safeRender(check.data_quality_check)}</td>
                  <td className="px-4 py-2 text-center">
                    <button
                      type="button"
                      onClick={() => handleEditClick(check)}
                      className="flex items-center px-2 py-1 text-xs font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700"
                      disabled={!!editingCheckId || isLoading}
                    >
                      <Edit2 size={16} className="mr-1" /> Edit
                    </button>
                  </td>
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

        {error && (
          <div className="my-3 p-3 bg-red-100 text-red-700 rounded break-all">
            {typeof error === 'object' ? safeRender(JSON.stringify(error)) : safeRender(error)}
          </div>
        )}
        
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