import React, { useState, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import { 
  MapPin, 
  Calendar, 
  Activity, 
  AlertTriangle, 
  Download, 
  Upload,
  Settings,
  Eye,
  MessageCircle,
  FileText,
  Bell,
  MoreVertical,
  RefreshCcw,
  VolumeX
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Brush
} from 'recharts';
import ScoreGauge from '../components/ScoreGauge';

function DeviceDetail() {
  const { id } = useParams();
  const [activeTab, setActiveTab] = useState('graphs');
  const [loading, setLoading] = useState(true);
  const [editMode, setEditMode] = useState(false);
  const [deviceData, setDeviceData] = useState(null);
  const [presiteCheckData, setPresiteCheckData] = useState(null);
  const [loadingPresiteCheck, setLoadingPresiteCheck] = useState(false);
  const [presiteCheckError, setPresiteCheckError] = useState(null);
  const [isPresiteEditMode, setIsPresiteEditMode] = useState(false);
  const [editablePresiteData, setEditablePresiteData] = useState(null);
  const [weeklyChecksData, setWeeklyChecksData] = useState(null);
  const [loadingWeeklyChecks, setLoadingWeeklyChecks] = useState(false);
  const [weeklyChecksError, setWeeklyChecksError] = useState(null);
  const [responsibilityData, setResponsibilityData] = useState([]);
  const [editableResponsibility, setEditableResponsibility] = useState(null);
  
  const fetchAllDeviceData = useCallback(async () => {
    if (!id) return;
    setLoading(true);
    try {
        const apiUrl = process.env.REACT_APP_API_BASE || '';
        const monitorPromise = fetch(`${apiUrl}/api/v1/monitors/${id}`);
        const responsibilityPromise = fetch(`${apiUrl}/api/v1/responsibilities/monitor/${id}`);

        const [monitorResponse, responsibilityResponse] = await Promise.all([monitorPromise, responsibilityPromise]);

        if (!monitorResponse.ok) throw new Error('Failed to fetch monitor data');
        if (!responsibilityResponse.ok && responsibilityResponse.status !== 404) {
             throw new Error('Failed to fetch responsibility data');
        }
        
        const monitorData = await monitorResponse.json();
        const respData = responsibilityResponse.ok ? await responsibilityResponse.json() : [];

        setDeviceData(monitorData);
        setResponsibilityData(respData);

    } catch (error) {
        console.error("Failed to fetch device details:", error);
    } finally {
        setLoading(false);
    }
  }, [id]);

  const fetchPresiteCheckData = useCallback(async () => {
    if (id) {
      try {
        setLoadingPresiteCheck(true);
        setPresiteCheckError(null);
        const apiUrl = process.env.REACT_APP_API_BASE || '';
        const response = await fetch(`${apiUrl}/api/v1/monitors/${id}/presite-check/latest`);
        if (!response.ok) {
          if (response.status === 404) {
            setPresiteCheckData(null); // Set to null if not found
          } else {
            throw new Error('Network response was not ok');
          }
        } else {
          const data = await response.json();
          setPresiteCheckData(data);
        }
      } catch (error) {
        setPresiteCheckError('Failed to load data.');
      } finally {
        setLoadingPresiteCheck(false);
      }
    }
  }, [id]);

  const fetchWeeklyChecksData = useCallback(async () => {
      if (id) {
        try {
          setLoadingWeeklyChecks(true);
          setWeeklyChecksError(null);
          const apiUrl = process.env.REACT_APP_API_BASE || '';
          const response = await fetch(`${apiUrl}/api/v1/monitors/${id}/weekly-quality-checks`);
          if (!response.ok) {
            throw new Error('Network response was not ok');
          } else {
            const data = await response.json();
            const groupedData = data.reduce((acc, check) => {
              const { interim, id, check_date, silt_mm, comments, actions } = check;
              if (!acc[interim]) acc[interim] = [];
              acc[interim].push({ id, check_date, silt: silt_mm, comment: comments, actions });
              return acc;
            }, {});
            setWeeklyChecksData(groupedData);
          }
        } catch (error) {
          setWeeklyChecksError('Failed to load data.');
        } finally {
          setLoadingWeeklyChecks(false);
        }
      }
  }, [id]);
  
  useEffect(() => {
    fetchAllDeviceData();
  }, [id, fetchAllDeviceData]);

  useEffect(() => {
    if (activeTab === 'arup-presite' || activeTab === 'install-checks') {
      fetchPresiteCheckData();
    } else if (activeTab === 'weekly-quality-check') {
      fetchWeeklyChecksData();
    }
  }, [activeTab, id, fetchPresiteCheckData, fetchWeeklyChecksData]);

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical': return 'bg-red-100 text-red-800';
      case 'warning': return 'bg-orange-100 text-orange-800';
      case 'info': return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'bg-red-100 text-red-800';
      case 'acknowledged': return 'bg-orange-100 text-orange-800';
      case 'resolved': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!deviceData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <p>Device not found.</p>
      </div>
    )
  }
  
  const responsibility = responsibilityData && responsibilityData.length > 0
    ? responsibilityData[0]
    : {};

  // Handlers for Presite/Install Checks
  const handlePresiteEdit = () => {
    setEditablePresiteData(presiteCheckData ? { ...presiteCheckData } : { 
        mh_reference: '', 
        pipe: '', 
        position: '',
        correct_location: true,
        correct_install_pipe: true,
        correct_pipe_size: true,
        correct_pipe_shape: true,
        comments: ''
    });
    setIsPresiteEditMode(true);
  };

  const handlePresiteCancel = () => {
    setIsPresiteEditMode(false);
    setEditablePresiteData(null);
  };

  const handlePresiteChange = (field, value) => {
    setEditablePresiteData(prev => ({...prev, [field]: value}));
  };
  
  const handleSavePresiteCheck = async () => {
    try {
        const apiUrl = process.env.REACT_APP_API_BASE || '';
        const payload = {
            ...editablePresiteData,
            monitor_id: deviceData.id,
        };
        // The id is from the old record, should not be in a create payload
        delete payload.id; 
        delete payload.checked_at;
        
        const response = await fetch(`${apiUrl}/api/v1/presite_install_check/`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        if(!response.ok) throw new Error("Failed to save presite check");
        
        setIsPresiteEditMode(false);
        setEditablePresiteData(null);
        fetchPresiteCheckData(); // Refetch data
    } catch (error) {
        console.error("Save failed:", error);
    }
  };

  // Handler for Weekly Check
  const handleSaveWeeklyCheck = async (updatedRow, interim) => {
    try {
        const apiUrl = process.env.REACT_APP_API_BASE || '';
        const method = updatedRow.id ? 'PUT' : 'POST';
        const endpoint = updatedRow.id 
            ? `${apiUrl}/api/v1/weekly_quality_check/${updatedRow.id}`
            : `${apiUrl}/api/v1/weekly_quality_check/`;

        const payload = {
            ...updatedRow,
            monitor_id: deviceData.id,
            check_date: updatedRow.check_date,
            silt_mm: updatedRow.silt,
            comments: updatedRow.comment,
            actions: updatedRow.actions,
            interim: interim,
        };
        // No need to send temporary frontend-only id to backend for creation
        if(!payload.id) delete payload.id; 

        const response = await fetch(endpoint, {
            method: method,
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });

        if(!response.ok) throw new Error("Failed to save weekly check");

        fetchWeeklyChecksData(); // Refetch to show updated data
        return true; // Indicate success
    } catch (error) {
        console.error("Save failed:", error);
        return false; // Indicate failure
    }
  };

  const handleAddNewWeeklyCheck = () => {
    const newCheck = {
        id: null, // Indicates a new record
        check_date: new Date().toISOString().split('T')[0],
        silt: '',
        comment: '',
        actions: '',
    };
    // This is tricky because data is grouped. We need to find the next interim number.
    const interimNumbers = Object.keys(weeklyChecksData || {}).map(k => parseInt(k.replace(/interim/i, ''))).filter(n => !isNaN(n));
    const nextInterimNum = interimNumbers.length > 0 ? Math.max(...interimNumbers) + 1 : 1;
    const newInterimName = `Interim${nextInterimNum}`;

    setWeeklyChecksData(prev => ({
        ...prev,
        [newInterimName]: [{ ...newCheck, isNew: true }] // Flag to auto-edit
    }));
  };

  const handleMainEdit = () => {
    const initialData = (responsibilityData && responsibilityData[0]) || {};
    let removalDateValue;

    if (initialData.id) {
        // Existing record
        removalDateValue = initialData.removal_date ? new Date(initialData.removal_date).toISOString().split('T')[0] : '';
    } else {
        // New record, default to today
        removalDateValue = new Date().toISOString().split('T')[0];
    }

    setEditableResponsibility({
        action_type: initialData.action_type || '',
        requester: initialData.requester || '',
        removal_checker: initialData.removal_checker || '',
        removal_reviewer: initialData.removal_reviewer || '',
        removal_date: removalDateValue,
        id: initialData.id || null,
    });
    setEditMode(true);
  };
  
  const handleMainCancel = () => {
    setEditMode(false);
    setEditableResponsibility(null);
  };
  
  const handleResponsibilityChange = (field, value) => {
    setEditableResponsibility(prev => ({...prev, [field]: value}));
  };

  const handleSaveResponsibility = async () => {
    const apiUrl = process.env.REACT_APP_API_BASE || '';
    const method = editableResponsibility.id ? 'PUT' : 'POST';
    const url = editableResponsibility.id 
        ? `${apiUrl}/api/v1/responsibilities/${editableResponsibility.id}`
        : `${apiUrl}/api/v1/responsibilities/`;
    
    const payload = { ...editableResponsibility };
    if (!editableResponsibility.id) {
        payload.monitor_id = deviceData.id;
    }
    delete payload.id;

    if (payload.removal_date === '') {
        payload.removal_date = null;
    }

    try {
        const response = await fetch(url, {
            method: method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) throw new Error("Failed to save responsibility data");
        
        await fetchAllDeviceData(); // Refetch all data
        setEditMode(false);
        setEditableResponsibility(null);

    } catch (error) {
        console.error("Save failed:", error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Page header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="py-4">
            {/* Breadcrumb navigation */}
            <nav className="flex mb-4" aria-label="Breadcrumb">
              <ol className="flex items-center space-x-2">
                <li>
                  <Link to="/" className="text-gray-500 hover:text-gray-700">Home</Link>
                </li>
                <li className="text-gray-500">/</li>
                <li>
                  <Link to="/fleet" className="text-gray-500 hover:text-gray-700">Fleet</Link>
                </li>
                <li className="text-gray-500">/</li>
                <li className="text-gray-900 font-medium">{deviceData.monitor_name}</li>
              </ol>
            </nav>

            {/* Device header info */}
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-4">
                  <div>
                    <h1 className="text-2xl font-bold text-gray-900">{deviceData.monitor_name}</h1>
                    <div className="flex items-center space-x-4 mt-1 text-sm text-gray-600">
                      <span>MH Reference: {deviceData.mh_reference}</span>
                      <div className="flex items-center space-x-1">
                        <MapPin className="w-4 h-4" />
                        <span>{deviceData.location}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Activity className="w-4 h-4" />
                        <span className="text-green-600">{deviceData.status === 'active' ? 'Online' : 'Offline'}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Quick action buttons */}
              <div className="flex items-center space-x-3 mt-4 lg:mt-0">
                <button className="flex items-center space-x-2 px-3 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700">
                  <RefreshCcw className="w-4 h-4" />
                  <span>Re-score</span>
                </button>
                <button className="flex items-center space-x-2 px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50">
                  <VolumeX className="w-4 h-4" />
                  <span>Mute Alerts</span>
                </button>
                <button className="flex items-center space-x-2 px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50">
                  <MoreVertical className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Left main content area */}
          <div className="lg:col-span-3">
            {/* Score gauge and KPIs */}
            <div className="bg-white rounded-lg shadow-md p-3 mb-6 relative">
                <div className="absolute top-3 right-3 z-10 flex space-x-2">
                    {editMode ? (
                        <>
                            <button
                                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm h-fit"
                                onClick={handleSaveResponsibility}
                            >
                                Save
                            </button>
                            <button
                                className="px-4 py-2 bg-gray-300 text-black rounded hover:bg-gray-400 text-sm h-fit"
                                onClick={handleMainCancel}
                            >
                                Cancel
                            </button>
                        </>
                    ) : (
                        <button
                            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm h-fit"
                            onClick={handleMainEdit}
                        >
                            Edit
                        </button>
                    )}
                </div>
                <table className="min-w-full divide-y divide-gray-200 text-sm w-full">
                  <tbody>
                    <tr>
                      <td className="font-semibold text-gray-700 pr-2 py-1">Install Date</td>
                      <td className="py-1">{deviceData.install_date ? new Date(deviceData.install_date).toLocaleDateString() : 'N/A'}</td>
                    </tr>
                    <tr>
                      <td className="font-semibold text-gray-700 pr-2 py-1">W3W</td>
                      <td className="py-1">{deviceData.w3w}</td>
                    </tr>
                    <tr>
                      <td className="font-semibold text-gray-700 pr-2 py-1">Pipe</td>
                      <td className="py-1">{deviceData.pipe}</td>
                    </tr>
                    <tr>
                      <td className="font-semibold text-gray-700 pr-2 py-1">Height (mm)</td>
                      <td className="py-1">{deviceData.height_mm}</td>
                    </tr>
                    <tr>
                      <td className="font-semibold text-gray-700 pr-2 py-1">Width (mm)</td>
                      <td className="py-1">{deviceData.width_mm}</td>
                    </tr>
                    <tr>
                      <td className="font-semibold text-gray-700 pr-2 py-1">Shape</td>
                      <td className="py-1">{deviceData.shape}</td>
                    </tr>
                    <tr>
                      <td className="font-semibold text-gray-700 pr-2 py-1">Depth (mm)</td>
                      <td className="py-1">{deviceData.depth_mm}</td>
                    </tr>
                    <tr>
                      <td className="font-semibold text-gray-700 pr-2 py-1">Actioned</td>
                      <td className="py-1 bg-yellow-100 font-semibold">
                        <input
                          type="text"
                          className={`w-full bg-yellow-100 font-semibold rounded px-2 py-1 ${editMode ? 'border border-gray-300' : 'border-none'}`}
                          value={editMode ? editableResponsibility.action_type : responsibility.action_type || 'N/A'}
                          onChange={(e) => handleResponsibilityChange('action_type', e.target.value)}
                          readOnly={!editMode}
                        />
                      </td>
                    </tr>
                    <tr>
                      <td className="font-semibold text-gray-700 pr-2 py-1">Requester</td>
                      <td className="py-1">
                        <input
                          type="text"
                          className={`w-full rounded px-2 py-1 ${editMode ? 'border border-gray-300' : 'border-none'}`}
                          value={editMode ? editableResponsibility.requester : responsibility.requester || ''}
                          onChange={(e) => handleResponsibilityChange('requester', e.target.value)}
                          readOnly={!editMode}
                        />
                      </td>
                    </tr>
                    <tr>
                      <td className="font-semibold text-gray-700 pr-2 py-1">Removal checker</td>
                      <td className="py-1">
                        <input
                          type="text"
                          className={`w-full rounded px-2 py-1 ${editMode ? 'border border-gray-300' : 'border-none'}`}
                          value={editMode ? editableResponsibility.removal_checker : responsibility.removal_checker || ''}
                          onChange={(e) => handleResponsibilityChange('removal_checker', e.target.value)}
                          readOnly={!editMode}
                        />
                      </td>
                    </tr>
                    <tr>
                      <td className="font-semibold text-gray-700 pr-2 py-1">Removal reviewer</td>
                      <td className="py-1">
                        <input
                          type="text"
                          className={`w-full rounded px-2 py-1 ${editMode ? 'border border-gray-300' : 'border-none'}`}
                          value={editMode ? editableResponsibility.removal_reviewer : responsibility.removal_reviewer || ''}
                          onChange={(e) => handleResponsibilityChange('removal_reviewer', e.target.value)}
                          readOnly={!editMode}
                        />
                      </td>
                    </tr>
                    <tr>
                      <td className="font-semibold text-gray-700 pr-2 py-1">Removal Date</td>
                      <td className="py-1 bg-yellow-100 font-semibold">
                        {editMode ? (
                          <input
                            type="date"
                            className={`w-full bg-yellow-100 font-semibold rounded px-2 py-1 border border-gray-300`}
                            value={editableResponsibility.removal_date}
                            onChange={(e) => handleResponsibilityChange('removal_date', e.target.value)}
                          />
                        ) : (
                          <span className="px-2 py-1">{responsibility.removal_date ? new Date(responsibility.removal_date).toLocaleDateString() : ''}</span>
                        )}
                      </td>
                    </tr>
                  </tbody>
                </table>
            </div>

            {/* Tab navigation */}
            <div className="bg-white rounded-lg shadow-md">
              <div className="border-b border-gray-200">
                <nav className="-mb-px flex space-x-8 px-6">
                  {[
                    { id: 'graphs', name: 'Graphs', icon: Eye },
                    { id: 'arup-presite', name: 'Arup Presite', icon: FileText },
                    { id: 'install-checks', name: 'Install Checks', icon: FileText },
                    { id: 'weekly-quality-check', name: 'Weekly quality check', icon: FileText },
                    { id: 'drydays-compliance', name: 'Drydays compliance', icon: FileText },
                    { id: 'storm-compliance', name: 'Storm compliance', icon: FileText },
                    { id: 'settings', name: 'Settings', icon: Settings }
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                        activeTab === tab.id
                          ? 'border-blue-500 text-blue-600'
                          : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                      }`}
                    >
                      <tab.icon className="w-4 h-4" />
                      <span>{tab.name}</span>
                    </button>
                  ))}
                </nav>
              </div>

              {/* Tab content */}
              <div className="p-6">
                {activeTab === 'graphs' && (
                  <div className="space-y-8">
                    {/* Rainfall intensity chart */}
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-2">Rainfall intensity (mm/hr)</h3>
                      <ResponsiveContainer width="100%" height={180}>
                        <LineChart data={[
                          { time: '00:00', value: 0 },
                          { time: '02:00', value: 5 },
                          { time: '04:00', value: 10 },
                          { time: '06:00', value: 20 },
                          { time: '08:00', value: 15 },
                          { time: '10:00', value: 0 }
                        ]}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis dataKey="time" stroke="#6b7280" />
                          <YAxis stroke="#6b7280" />
                          <Tooltip />
                          <Line type="monotone" dataKey="value" stroke="#2563eb" strokeWidth={2} name="Rainfall" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    {/* Depth chart */}
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-2">Depth (m)</h3>
                      <ResponsiveContainer width="100%" height={180}>
                        <LineChart data={[
                          { time: '00:00', value: 0.05 },
                          { time: '02:00', value: 0.08 },
                          { time: '04:00', value: 0.12 },
                          { time: '06:00', value: 0.09 },
                          { time: '08:00', value: 0.11 },
                          { time: '10:00', value: 0.10 }
                        ]}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis dataKey="time" stroke="#6b7280" />
                          <YAxis stroke="#6b7280" />
                          <Tooltip />
                          <Line type="monotone" dataKey="value" stroke="#ef4444" strokeWidth={2} name="Depth" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    {/* Flow chart */}
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-2">Flow (m3/s)</h3>
                      <ResponsiveContainer width="100%" height={180}>
                        <LineChart data={[
                          { time: '00:00', value: 0.001 },
                          { time: '02:00', value: 0.002 },
                          { time: '04:00', value: 0.003 },
                          { time: '06:00', value: 0.0025 },
                          { time: '08:00', value: 0.002 },
                          { time: '10:00', value: 0.0015 }
                        ]}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis dataKey="time" stroke="#6b7280" />
                          <YAxis stroke="#6b7280" />
                          <Tooltip />
                          <Line type="monotone" dataKey="value" stroke="#22c55e" strokeWidth={2} name="Flow" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    {/* Velocity chart */}
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-2">Velocity (m/s)</h3>
                      <ResponsiveContainer width="100%" height={180}>
                        <LineChart data={[
                          { time: '00:00', value: 0.1 },
                          { time: '02:00', value: 0.15 },
                          { time: '04:00', value: 0.2 },
                          { time: '06:00', value: 0.18 },
                          { time: '08:00', value: 0.13 },
                          { time: '10:00', value: 0.11 }
                        ]}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis dataKey="time" stroke="#6b7280" />
                          <YAxis stroke="#6b7280" />
                          <Tooltip />
                          <Line type="monotone" dataKey="value" stroke="#a21caf" strokeWidth={2} name="Velocity" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                {activeTab === 'arup-presite' && (
                  <div className="space-y-6">
                    <div className="flex justify-between items-center">
                      <h3 className="text-lg font-semibold text-gray-900">Arup Presite</h3>
                      <div>
                        {isPresiteEditMode ? (
                          <div className="flex space-x-2">
                            <button onClick={handleSavePresiteCheck} className="px-3 py-1 bg-blue-600 text-white rounded text-sm">Save</button>
                            <button onClick={handlePresiteCancel} className="px-3 py-1 bg-gray-300 text-black rounded text-sm">Cancel</button>
                          </div>
                        ) : (
                          <button onClick={handlePresiteEdit} className="px-3 py-1 bg-blue-600 text-white rounded text-sm">Edit</button>
                        )}
                      </div>
                    </div>
                    {loadingPresiteCheck ? (
                      <p>Loading...</p>
                    ) : presiteCheckError ? (
                      <p className="text-red-500">{presiteCheckError}</p>
                    ) : isPresiteEditMode ? (
                      <table className="min-w-full divide-y divide-gray-200 text-sm">
                        <thead className="bg-gray-50">
                          <tr>
                            <th className="px-4 py-2 text-left font-semibold text-gray-700">MH Reference</th>
                            <th className="px-4 py-2 text-left font-semibold text-gray-700">Pipe</th>
                            <th className="px-4 py-2 text-left font-semibold text-gray-700">Position (inc/out)</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td><input className="w-full border rounded p-1" value={editablePresiteData.mh_reference} onChange={e => handlePresiteChange('mh_reference', e.target.value)} /></td>
                            <td><input className="w-full border rounded p-1" value={editablePresiteData.pipe} onChange={e => handlePresiteChange('pipe', e.target.value)} /></td>
                            <td><input className="w-full border rounded p-1" value={editablePresiteData.position} onChange={e => handlePresiteChange('position', e.target.value)} /></td>
                          </tr>
                        </tbody>
                      </table>
                    ) : presiteCheckData ? (
                      <table className="min-w-full divide-y divide-gray-200 text-sm">
                        <thead className="bg-gray-50">
                          <tr>
                            <th className="px-4 py-2 text-left font-semibold text-gray-700">MH Reference</th>
                            <th className="px-4 py-2 text-left font-semibold text-gray-700">Pipe</th>
                            <th className="px-4 py-2 text-left font-semibold text-ray-700">Position (inc/out)</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td className="px-4 py-2">{presiteCheckData.mh_reference}</td>
                            <td className="px-4 py-2">{presiteCheckData.pipe}</td>
                            <td className="px-4 py-2">{presiteCheckData.position}</td>
                          </tr>
                        </tbody>
                      </table>
                    ) : (
                      <p>No data available.</p>
                    )}
                  </div>
                )}

                {activeTab === 'install-checks' && (
                  <div className="space-y-6">
                    <div className="flex justify-between items-center">
                      <h3 className="text-lg font-semibold text-gray-900">Install Checks</h3>
                      <div>
                        {isPresiteEditMode ? (
                          <div className="flex space-x-2">
                            <button onClick={handleSavePresiteCheck} className="px-3 py-1 bg-blue-600 text-white rounded text-sm">Save</button>
                            <button onClick={handlePresiteCancel} className="px-3 py-1 bg-gray-300 text-black rounded text-sm">Cancel</button>
                          </div>
                        ) : (
                          <button onClick={handlePresiteEdit} className="px-3 py-1 bg-blue-600 text-white rounded text-sm">Edit</button>
                        )}
                      </div>
                    </div>
                    {loadingPresiteCheck ? (
                      <p>Loading...</p>
                    ) : presiteCheckError ? (
                      <p className="text-red-500">{presiteCheckError}</p>
                    ) : isPresiteEditMode ? (
                      <table className="min-w-full divide-y divide-gray-200 text-sm">
                        {/* Editable table */}
                        <thead className="bg-gray-50">
                          <tr>
                            <th className="px-4 py-2 text-left font-semibold text-gray-700">Correct Location?</th>
                            <th className="px-4 py-2 text-left font-semibold text-gray-700">Correct Install Pipe?</th>
                            <th className="px-4 py-2 text-left font-semibold text-gray-700">Correct Pipe Size?</th>
                            <th className="px-4 py-2 text-left font-semibold text-gray-700">Correct Pipe Shape?</th>
                            <th className="px-4 py-2 text-left font-semibold text-gray-700">Comments</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td>
                              <select className="w-full border rounded p-1" value={editablePresiteData.correct_location ? 'Y' : 'N'} onChange={e => handlePresiteChange('correct_location', e.target.value === 'Y')}>
                                <option value="Y">Y</option>
                                <option value="N">N</option>
                              </select>
                            </td>
                            <td>
                              <select className="w-full border rounded p-1" value={editablePresiteData.correct_install_pipe ? 'Y' : 'N'} onChange={e => handlePresiteChange('correct_install_pipe', e.target.value === 'Y')}>
                                <option value="Y">Y</option>
                                <option value="N">N</option>
                              </select>
                            </td>
                            <td>
                              <select className="w-full border rounded p-1" value={editablePresiteData.correct_pipe_size ? 'Y' : 'N'} onChange={e => handlePresiteChange('correct_pipe_size', e.target.value === 'Y')}>
                                <option value="Y">Y</option>
                                <option value="N">N</option>
                              </select>
                            </td>
                            <td>
                              <select className="w-full border rounded p-1" value={editablePresiteData.correct_pipe_shape ? 'Y' : 'N'} onChange={e => handlePresiteChange('correct_pipe_shape', e.target.value === 'Y')}>
                                <option value="Y">Y</option>
                                <option value="N">N</option>
                              </select>
                            </td>
                            <td><input className="w-full border rounded p-1" value={editablePresiteData.comments || ''} onChange={e => handlePresiteChange('comments', e.target.value)} /></td>
                          </tr>
                        </tbody>
                      </table>
                    ) : presiteCheckData ? (
                      <table className="min-w-full divide-y divide-gray-200 text-sm">
                        {/* View table */}
                        <thead className="bg-gray-50">
                          <tr>
                            <th className="px-4 py-2 text-left font-semibold text-gray-700">Correct Location?</th>
                            <th className="px-4 py-2 text-left font-semibold text-gray-700">Correct Install Pipe?</th>
                            <th className="px-4 py-2 text-left font-semibold text-gray-700">Correct Pipe Size?</th>
                            <th className="px-4 py-2 text-left font-semibold text-gray-700">Correct Pipe Shape?</th>
                            <th className="px-4 py-2 text-left font-semibold text-gray-700">Comments</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td className="px-4 py-2">{presiteCheckData.correct_location ? 'Y' : 'N'}</td>
                            <td className="px-4 py-2">{presiteCheckData.correct_install_pipe ? 'Y' : 'N'}</td>
                            <td className="px-4 py-2">{presiteCheckData.correct_pipe_size ? 'Y' : 'N'}</td>
                            <td className="px-4 py-2">{presiteCheckData.correct_pipe_shape ? 'Y' : 'N'}</td>
                            <td className="px-4 py-2">{presiteCheckData.comments ? JSON.stringify(presiteCheckData.comments) : ''}</td>
                          </tr>
                        </tbody>
                      </table>
                    ) : (
                      <p>No data available.</p>
                    )}
                  </div>
                )}

                {activeTab === 'weekly-quality-check' && (
                  <div className="space-y-8 overflow-x-auto">
                    <div className="flex justify-between items-center mb-4">
                      <h3 className="text-lg font-semibold text-gray-900">Weekly quality check</h3>
                      <button onClick={handleAddNewWeeklyCheck} className="px-3 py-1 bg-green-600 text-white rounded text-sm">Add New</button>
                    </div>
                    {loadingWeeklyChecks ? (
                      <p>Loading...</p>
                    ) : weeklyChecksError ? (
                      <p className="text-red-500">{weeklyChecksError}</p>
                    ) : weeklyChecksData && Object.keys(weeklyChecksData).length > 0 ? (
                      Object.entries(weeklyChecksData).map(([interimName, rows]) => (
                        <WeeklyQualityTable
                          key={interimName}
                          interim={interimName}
                          rows={rows}
                          onSave={handleSaveWeeklyCheck}
                        />
                      ))
                    ) : (
                      <p>No weekly quality check data available.</p>
                    )}
                  </div>
                )}

                {activeTab === 'drydays-compliance' && (
                  <div className="space-y-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Drydays compliance</h3>
                    <table className="min-w-full divide-y divide-gray-200 text-sm">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-4 py-2 text-left font-semibold text-gray-700">DryDay</th>
                          <th className="px-4 py-2 text-left font-semibold text-gray-700">Date</th>
                          <th className="px-4 py-2 text-left font-semibold text-gray-700">COMMENT</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td className="px-4 py-2">DryDay 1</td>
                          <td className="px-4 py-2">10/08/2024</td>
                          <td className="px-4 py-2">Poor data.Velocity data spiky.</td>
                        </tr>
                        <tr>
                          <td className="px-4 py-2">DryDay 2</td>
                          <td className="px-4 py-2">2024/8/17</td>
                          <td className="px-4 py-2">N - Poor Data</td>
                        </tr>
                        <tr>
                          <td className="px-4 py-2">DryDay 3</td>
                          <td className="px-4 py-2">2024/9/4</td>
                          <td className="px-4 py-2">N - Poor Data</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                )}

                {activeTab === 'storm-compliance' && (
                  <div className="space-y-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Storm compliance</h3>
                    <table className="min-w-full divide-y divide-gray-200 text-sm">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-4 py-2 text-left font-semibold text-gray-700">Storm</th>
                          <th className="px-4 py-2 text-left font-semibold text-gray-700">Date</th>
                          <th className="px-4 py-2 text-left font-semibold text-gray-700">COMMENT</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td className="px-4 py-2">storm A</td>
                          <td className="px-4 py-2">23/08/2024</td>
                          <td className="px-4 py-2">Y</td>
                        </tr>
                        <tr>
                          <td className="px-4 py-2">storm B</td>
                          <td className="px-4 py-2">2024/9/1</td>
                          <td className="px-4 py-2">Y</td>
                        </tr>
                        <tr>
                          <td className="px-4 py-2">storm C</td>
                          <td className="px-4 py-2">2024/9/6</td>
                          <td className="px-4 py-2">N - No Data</td>
                        </tr>
                      </tbody>
                    </table>
                    <table className="min-w-full divide-y divide-gray-200 text-sm mt-8">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-4 py-2 text-left font-semibold text-gray-700 w-3/4">Comment</th>
                          <th className="px-4 py-2 text-left font-semibold text-gray-700 w-1/4">Storm Coverage</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td className="px-4 py-2">All storms covered, data quality good.</td>
                          <td className="px-4 py-2">3+</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                )}

                {activeTab === 'settings' && (
                  <div className="space-y-6">
                    <h3 className="text-lg font-semibold text-gray-900">Device Settings</h3>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Device Name
                        </label>
                        <input
                          type="text"
                          value={deviceData.monitor_name}
                          className="w-full border border-gray-300 rounded-md px-3 py-2"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          External ID
                        </label>
                        <input
                          type="text"
                          value={deviceData.mh_reference}
                          className="w-full border border-gray-300 rounded-md px-3 py-2"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Installation Date
                        </label>
                        <input
                          type="date"
                          value={deviceData.install_date ? new Date(deviceData.install_date).toISOString().split('T')[0] : ''}
                          className="w-full border border-gray-300 rounded-md px-3 py-2"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Location
                        </label>
                        <input
                          type="text"
                          value={deviceData.location}
                          className="w-full border border-gray-300 rounded-md px-3 py-2"
                        />
                      </div>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Maintenance Notes
                      </label>
                      <textarea
                        rows="4"
                        value={''} // a field for this is not in the model yet
                        className="w-full border border-gray-300 rounded-md px-3 py-2"
                      ></textarea>
                    </div>
                    
                    <div className="flex justify-between items-center pt-6 border-t border-gray-200">
                      <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">
                        Save Changes
                      </button>
                      
                      <button className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700">
                        Delete Device
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right sidebar */}
          <div className="space-y-6">
            {/* Storm A */}
            <div className="bg-white rounded-lg shadow-md p-4">
              <h4 className="font-semibold text-gray-900 mb-3">Storm A</h4>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Date</span>
                  <span className="text-sm font-medium">23/08/2024</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Average depth (mm)</span>
                  <span className="text-sm font-medium">16.76</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Average peak intensity (mm/hr) </span>
                  <span className="text-sm font-medium">33.00</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Average duration (min)</span>
                  <span className="text-sm font-medium">21</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">RG Qualified</span>
                  <span className="text-sm font-medium">87%</span>
                </div>                
              </div>
            </div>

            {/* Storm B */}
            <div className="bg-white rounded-lg shadow-md p-4">
              <h4 className="font-semibold text-gray-900 mb-3">Storm B</h4>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Date</span>
                  <span className="text-sm font-medium">2024/9/1</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Average depth (mm)</span>
                  <span className="text-sm font-medium">25.32</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Average peak intensity (mm/hr) </span>
                  <span className="text-sm font-medium">33.69</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Average duration (min)</span>
                  <span className="text-sm font-medium">53</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">RG Qualified</span>
                  <span className="text-sm font-medium">81%</span>
                </div>                
              </div>
            </div>
            {/* Storm B */}
            <div className="bg-white rounded-lg shadow-md p-4">
              <h4 className="font-semibold text-gray-900 mb-3">Storm C</h4>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Date</span>
                  <span className="text-sm font-medium">2024/9/6</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Average depth (mm)</span>
                  <span className="text-sm font-medium">38.88</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Average peak intensity (mm/hr) </span>
                  <span className="text-sm font-medium">69.23</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Average duration (min)</span>
                  <span className="text-sm font-medium">33</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">RG Qualified</span>
                  <span className="text-sm font-medium">81%</span>
                </div>                
              </div>
            </div>            
          </div>
        </div>
      </div>
    </div>
  );
}

function WeeklyQualityTable({ interim, rows, onSave }) {
  const [editIdx, setEditIdx] = React.useState(null);
  const [editRows, setEditRows] = React.useState(rows);

  useEffect(() => {
    // If a new row is added, automatically set it to edit mode.
    const newRowIndex = rows.findIndex(r => r.isNew);
    if (newRowIndex !== -1) {
      setEditIdx(newRowIndex);
    }
  }, [rows]);

  const handleEdit = (idx) => setEditIdx(idx);
  const handleDone = () => setEditIdx(null);
  const handleChange = (idx, field, value) => {
    setEditRows(rws => rws.map((r, i) => i === idx ? { ...r, [field]: value } : r));
  };
  const handleSave = async (idx) => {
    const rowToSave = { ...editRows[idx] };
    delete rowToSave.isNew; // Clean up the flag before saving
    const success = await onSave(rowToSave, interim);
    if (success) {
      setEditIdx(null);
    }
  };
  return (
    <table className="w-full table-fixed min-w-[900px] divide-y divide-gray-200 text-sm">
      <thead className="bg-gray-50">
        <tr>
          <th className="px-4 py-2 text-left font-semibold text-gray-700 w-20">{interim}</th>
          <th className="px-4 py-2 text-left font-semibold text-gray-700 w-24">Check Date</th>
          <th className="px-4 py-2 text-left font-semibold text-gray-700 w-24">Silt(mm)</th>
          <th className="px-4 py-2 text-left font-semibold text-gray-700 w-2/5">COMMENT</th>
          <th className="px-4 py-2 text-left font-semibold text-gray-700 w-1/5">ACTIONS</th>
          <th className="px-4 py-2 text-left font-semibold text-gray-700 w-20"></th>
        </tr>
      </thead>
      <tbody>
        {editRows.map((row, idx) => (
          <tr key={idx}>
            <td className="px-4 py-2">{interim.toLowerCase()}</td>
            <td className="px-4 py-2">
              {editIdx === idx ? (
                <input type="date" className="border rounded px-1 py-0.5 w-full" value={row.check_date ? new Date(row.check_date).toISOString().split('T')[0] : ''} onChange={e => handleChange(idx, 'check_date', e.target.value)} />
              ) : (row.check_date ? new Date(row.check_date).toLocaleDateString() : 'N/A')}
            </td>
            <td className="px-4 py-2">
              {editIdx === idx ? (
                <input className="border rounded px-1 py-0.5 w-full" value={row.silt} onChange={e => handleChange(idx, 'silt', e.target.value)} />
              ) : row.silt}
            </td>
            <td className="px-4 py-2">
              {editIdx === idx ? (
                <input className="border rounded px-1 py-0.5 w-full" value={row.comment} onChange={e => handleChange(idx, 'comment', e.target.value)} />
              ) : row.comment}
            </td>
            <td className="px-4 py-2">
              {editIdx === idx ? (
                <input className="border rounded px-1 py-0.5 w-full" value={row.actions} onChange={e => handleChange(idx, 'actions', e.target.value)} />
              ) : row.actions}
            </td>
            <td className="px-4 py-2">
              {editIdx === idx ? (
                <div className="flex space-x-2">
                  <button onClick={() => handleSave(idx)} className="px-2 py-1 bg-blue-600 text-white rounded text-xs">Save</button>
                  <button onClick={handleDone} className="px-2 py-1 bg-gray-300 text-black rounded text-xs">Cancel</button>
                </div>
              ) : (
                <button className="px-2 py-1 bg-blue-600 text-white rounded text-xs" onClick={() => handleEdit(idx)}>Edit</button>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default DeviceDetail; 