import React, { useState, useEffect, useCallback } from 'react';
import { X } from 'lucide-react';
import axios from 'axios';
import CheckRow from './CheckRow';

const API_BASE_URL = process.env.REACT_APP_API_BASE || 'http://localhost:8000'; // Hardcoded for now

const PresiteInstallCheckModal = ({ monitors, onClose }) => {
  const [checksData, setChecksData] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchLatestChecks = useCallback(async () => {
    if (!monitors || monitors.length === 0) {
      setChecksData([]);
      return;
    }

    setIsLoading(true);
    setError('');
    try {
      const monitorIds = monitors.map(m => m.id);
      const response = await axios.post(`${API_BASE_URL}/api/v1/presite_install_check/latest_batch`, monitorIds);
      const latestChecks = response.data;
      
      const checksMap = latestChecks.reduce((acc, check) => {
        acc[check.monitor_id] = check;
        return acc;
      }, {});

      const mergedData = monitors.map(monitor => {
        const latestCheck = checksMap[monitor.id];
        if (latestCheck) {
          return {
            ...latestCheck,
            monitor_name: monitor.monitor_name,
            correct_location: latestCheck.correct_location ? 'Y' : 'N',
            correct_install_pipe: latestCheck.correct_install_pipe ? 'Y' : 'N',
            correct_pipe_size: latestCheck.correct_pipe_size ? 'Y' : 'N',
            correct_pipe_shape: latestCheck.correct_pipe_shape ? 'Y' : 'N',
            comments: latestCheck.comments?.notes || '',
          };
        }
        return {
          monitor_id: monitor.id,
          monitor_name: monitor.monitor_name,
          mh_reference: '',
          pipe: '',
          position: '',
          correct_location: 'Y',
          correct_install_pipe: 'Y',
          correct_pipe_size: 'Y',
          correct_pipe_shape: 'Y',
          comments: '',
        };
      });
      setChecksData(mergedData);
    } catch (err) {
      console.error("Error fetching latest presite checks:", err);
      setError('Failed to load latest check data.');
    } finally {
      setIsLoading(false);
    }
  }, [monitors]);

  useEffect(() => {
    fetchLatestChecks();
  }, [fetchLatestChecks]);

  const handleInputChange = useCallback((monitorId, field, value) => {
    setChecksData(prev =>
      prev.map(check =>
        check.monitor_id === monitorId ? { ...check, [field]: value } : check
      )
    );
  }, []);
  
  const handleSubmit = async () => {
    setIsLoading(true);
    setError('');

    const payloads = checksData.map(check => {
      const payload = { ...check };
      delete payload.monitor_name;
      delete payload.id;
      delete payload.checked_at;
      
      return {
        ...payload,
        correct_location: payload.correct_location === 'Y',
        correct_install_pipe: payload.correct_install_pipe === 'Y',
        correct_pipe_size: payload.correct_pipe_size === 'Y',
        correct_pipe_shape: payload.correct_pipe_shape === 'Y',
      };
    });

    if (error) {
        setIsLoading(false);
        return;
    }

    try {
        await axios.post(`${API_BASE_URL}/api/v1/presite_install_check/batch`, payloads);
        onClose();
    } catch (err) {
        console.error("Error saving presite checks:", err);
        setError(err.response?.data?.detail || 'Failed to save checks. Please try again.');
    } finally {
        setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
      <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-7xl max-h-[90vh] flex flex-col">
        <div className="flex justify-between items-center border-b pb-3">
          <h2 className="text-xl font-bold text-gray-800">Presite & Install Check</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
            <X size={24} />
          </button>
        </div>

        {error && <div className="my-3 p-3 bg-red-100 text-red-700 rounded">{error}</div>}
        
        <div className="flex-grow overflow-y-auto mt-4">
          <table className="min-w-full divide-y divide-gray-200 table-fixed">
            <thead className="bg-gray-50 sticky top-0">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[12%]">FM/DM/PL</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[8%]">MH Reference</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[8%]">Pipe</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[8%]">Position</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[8%]">Correct Location</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[8%]">Correct Install Pipe</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[8%]">Correct Pipe Size</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[8%]">Correct Pipe Shape</th>
                <th className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-[32%]">Comments</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {isLoading && !checksData.length ? (
                <tr><td colSpan="9" className="text-center py-4">Loading checks...</td></tr>
              ) : checksData.map(check => (
                <CheckRow 
                  key={check.monitor_id} 
                  check={check}
                  handleInputChange={handleInputChange} 
                />
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex-shrink-0 flex justify-end mt-4">
          <button 
            type="button" 
            onClick={handleSubmit} 
            disabled={isLoading} 
            className="px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:bg-gray-400"
          >
            {isLoading ? 'Saving...' : 'Save All'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default PresiteInstallCheckModal; 