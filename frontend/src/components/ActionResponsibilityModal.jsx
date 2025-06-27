import React, { useState, useEffect, useCallback } from 'react';
import { X } from 'lucide-react';
import ResponsibilityCheckRow from './ResponsibilityCheckRow';

function ActionResponsibilityModal({ isOpen, onClose, selectedMonitors, onSave }) {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [responsibilities, setResponsibilities] = useState([]);

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
                    action_type: '',
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
                item.monitor_id === monitorId ? { ...item, [field]: value } : item
            )
        );
    }, []);
    
    const handleSave = async () => {
        setLoading(true);
        setError(null);

        const toUpdate = responsibilities
            .filter(item => item.id)
            .map(({ monitorName, monitor_id, ...rest }) => rest);

        const toCreate = responsibilities
            .filter(item => 
                !item.id && (item.action_type || item.requester || item.removal_checker || item.removal_reviewer || item.removal_date)
            )
            .map(({ id, monitorName, ...rest }) => rest);

        try {
            await onSave({ toUpdate, toCreate });
            onClose();
        } catch (err) {
            setError(err.message || 'Failed to save changes.');
        } finally {
            setLoading(false);
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
                    <button 
                        onClick={handleSave} 
                        disabled={loading} 
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