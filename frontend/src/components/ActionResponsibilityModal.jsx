import React, { useState, useEffect, useCallback } from 'react';
import { X } from 'lucide-react';
import ResponsibilityCheckRow from './ResponsibilityCheckRow';

function ActionResponsibilityModal({ isOpen, onClose, selectedMonitors, onSave }) {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [responsibilities, setResponsibilities] = useState([]);
    // Add AI progress state
    const [isAIPredicting, setIsAIPredicting] = useState(false);
    const [aiProgress, setAiProgress] = useState({ current: 0, total: 1 });

    const fetchResponsibilities = useCallback(async () => {
        if (!isOpen || !selectedMonitors || selectedMonitors.length === 0) {
            setResponsibilities([]);
            return;
        }

        setLoading(true);
        setError(null);
        try {
            const apiUrl = process.env.REACT_APP_API_BASE || '';
            const monitorIds = selectedMonitors.map(m => m.id);
            const response = await fetch(`${apiUrl}/api/v1/responsibilities/bulk_get`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(monitorIds)
            });

            if (!response.ok) throw new Error('Failed to fetch responsibility data.');
            
            const fetchedData = await response.json();
            
            // Create a map for quick lookup
            const responsibilitiesMap = fetchedData.reduce((acc, resp) => {
                acc[resp.monitor_id] = resp;
                return acc;
            }, {});

            // Sort and merge based on the order of selectedMonitors
            const processedData = selectedMonitors.map(monitor => {
                const responsibility = responsibilitiesMap[monitor.id];
                if (responsibility) {
                    return { ...responsibility, monitorName: monitor.monitor_name || 'N/A' };
                }
                // Create a new empty responsibility if one doesn't exist
                return {
                    monitor_id: monitor.id,
                    monitorName: monitor.monitor_name || 'N/A',
                    action: '',
                    requester: '',
                    removal_checker: '',
                    removal_reviewer: '',
                    removal_date: null,
                };
            });
            
            setResponsibilities(processedData);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [isOpen, selectedMonitors]);

    useEffect(() => {
        fetchResponsibilities();
    }, [fetchResponsibilities]);

    const handleFieldChange = useCallback((monitorId, field, value) => {
        setResponsibilities(prev =>
            prev.map(item =>
                item.monitor_id === monitorId
                    ? {
                        ...item,
                        [field]: field === 'actions' ? wrapIfString(value, 'actions') : value
                    }
                    : item
            )
        );
    }, []);
    
    // Utility to wrap string as object for JSON fields
    const wrapIfString = (val, key) => {
      if (val == null) return null;
      if (typeof val === 'string') {
        if (val.trim() === '') return null;
        if (key === 'actions') return { actions: val };
      }
      return val;
    };

    const handleSave = async () => {
        setLoading(true);
        setError(null);

        // In handleSave, before sending to backend, wrap actions field for both toUpdate and toCreate
        const toUpdate = responsibilities
            .filter(item => item.id)
            .map(({ monitorName, ...rest }) => ({
                ...rest,
                actions: wrapIfString(rest.actions, 'actions'),
            }));

        const toCreate = responsibilities
            .filter(item => 
                !item.id && (item.action || item.requester || item.removal_checker || item.removal_reviewer || item.removal_date)
            )
            .map(({ id, monitorName, ...rest }) => ({
                ...rest,
                actions: wrapIfString(rest.actions, 'actions'),
            }));

        try {
            await onSave({ toUpdate, toCreate });
            onClose();
        } catch (err) {
            setError(err.message || 'Failed to save changes.');
        } finally {
            setLoading(false);
        }
    };

    // Add AI Predict Action handler
    const handleAIPredict = async () => {
        setIsAIPredicting(true);
        setAiProgress({ current: 0, total: 1 });
        setError(null);
        try {
            const apiUrl = process.env.REACT_APP_API_BASE || '';
            const monitorIds = selectedMonitors.map(m => m.monitor_id || m.id || m);
            // Improved progress bar animation: max 95% until done
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += 0.03; // 3% per tick
                if (progress < 0.95) {
                    setAiProgress({ current: progress, total: 1 });
                }
            }, 200);
            const response = await fetch(`${apiUrl}/api/v1/ai/predict-action`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(monitorIds)
            });
            clearInterval(progressInterval);
            setAiProgress({ current: 1, total: 1 }); // jump to 100%
            if (!response.ok) throw new Error('AI prediction failed.');
            const data = await response.json();
            // Check for missing data or errors in results
            const missing = (data.results || []).filter(r => r.error);
            if (missing.length > 0) {
                alert(missing.map(m => `${m.monitor_id}: ${m.error}`).join('\n'));
            } else {
                await fetchResponsibilities();
            }
        } catch (err) {
            setError(err.message || 'AI prediction failed.');
        } finally {
            setIsAIPredicting(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex justify-center items-center">
            <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-7xl max-h-[90vh] flex flex-col">
                <div className="flex justify-between items-center border-b pb-3 mb-4">
                    <h2 className="text-xl font-bold text-gray-800">Edit Action Responsibilities</h2>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
                        <X size={24} />
                    </button>
                </div>
                {/* AI Predict Progress Bar */}
                {isAIPredicting && (
                  <div className="w-full flex items-center space-x-2 my-2">
                    <div className="flex-1 h-2 bg-gray-200 rounded">
                      <div
                        className="h-2 bg-blue-500 rounded"
                        style={{ width: `${Math.min((aiProgress.current / aiProgress.total) * 100, 100)}%` }}
                      ></div>
                    </div>
                    <span className="text-sm text-gray-600 min-w-max">{`AI predicting...`}</span>
                  </div>
                )}
                {loading && <p className="text-center">Loading...</p>}
                {error && <div className="my-3 p-3 bg-red-100 text-red-700 rounded text-center">{error}</div>}

                <div className="overflow-y-auto flex-grow">
                    <table className="min-w-full divide-y divide-gray-200 text-sm">
                        <thead className="bg-gray-50 sticky top-0">
                            <tr>
                                <th className="px-4 py-2 text-left font-semibold text-gray-700 w-[15%]">FM/DM/PL</th>
                                <th className="px-4 py-2 text-left font-semibold text-gray-700 w-[15%]">Actioned</th>
                                <th className="px-4 py-2 text-left font-semibold text-gray-700 w-[15%]">Requester</th>
                                <th className="px-4 py-2 text-left font-semibold text-gray-700 w-[15%]">Removal Checker</th>
                                <th className="px-4 py-2 text-left font-semibold text-gray-700 w-[15%]">Removal Reviewer</th>
                                <th className="px-4 py-2 text-left font-semibold text-gray-700 w-[25%]">Removal Date</th>
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                            {responsibilities.map(resp => (
                                <ResponsibilityCheckRow
                                    key={resp.monitor_id}
                                    responsibility={resp}
                                    onFieldChange={handleFieldChange}
                                />
                            ))}
                        </tbody>
                    </table>
                </div>

                <div className="pt-4 mt-auto flex justify-end space-x-2">
                    {/* AI Predict Action button - same style as Save All */}
                    <button 
                        onClick={handleAIPredict}
                        disabled={loading || isAIPredicting} 
                        className="px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:bg-gray-400"
                    >
                        AI Predict Action
                    </button>
                    <button 
                        onClick={handleSave} 
                        disabled={loading || isAIPredicting} 
                        className="px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:bg-gray-400"
                    >
                        {loading ? 'Saving...' : 'Save All'}
                    </button>
                </div>
            </div>
        </div>
    );
}

export default ActionResponsibilityModal; 