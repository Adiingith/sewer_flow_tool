import React, { useState, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import { 
  MapPin, 
  Activity, 
  Settings,
  Eye,
  FileText,
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
  ResponsiveContainer
} from 'recharts';
import dayjs from 'dayjs';


// Custom Tooltip component for charts
const CustomTooltip = ({ active, payload, label, title, unit }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const timestamp = data.timestamp;
    // Use UTC time to avoid timezone conversion issues
    const time = timestamp ? new Date(timestamp).toISOString().slice(0, 19).replace('T', ' ') : '';
    
    return (
      <div className="bg-white p-2 border border-gray-300 rounded shadow-lg">
        <p className="font-medium text-gray-900">Time: {time} UTC</p>
        <p className="text-blue-600">{title}: {data.value} {unit}</p>
      </div>
    );
  }
  return null;
};

// Utility to wrap string as object for JSON fields
const wrapIfString = (val, key) => {
  if (val == null) return null;
  if (typeof val === 'string') {
    if (val.trim() === '') return null;
    if (key === 'actions') return { actions: val };
    if (key === 'comments') return { notes: val };
  }
  return val;
};

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
  const [measurementData, setMeasurementData] = useState([]);
  const [rainGaugeData, setRainGaugeData] = useState([]);
  const [startTime, setStartTime] = useState('');
  const [endTime, setEndTime] = useState('');
  const [minTime, setMinTime] = useState('');
  const [maxTime, setMaxTime] = useState('');
  const [initialLoading, setInitialLoading] = useState(true);
  const [statsData, setStatsData] = useState(null);
  const [stormEvents, setStormEvents] = useState([]);
  const [availableInterims, setAvailableInterims] = useState(null);
  const [selectedInterim, setSelectedInterim] = useState(null);
  const [loadingInterims, setLoadingInterims] = useState(false);
  const [latestWeeklyAction, setLatestWeeklyAction] = useState(null);
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

  const fetchAvailableInterims = useCallback(async () => {
    if (!id) return;
    setLoadingInterims(true);
    try {
      const apiUrl = process.env.REACT_APP_API_BASE || '';
      
      // Check if this is an RG device by checking deviceData
      const isRGDevice = deviceData && deviceData.monitor_name && deviceData.monitor_name.startsWith('RG');
      
      let endpoint;
      if (isRGDevice) {
        endpoint = `${apiUrl}/api/v1/monitors/${id}/rg-available-interims`;
      } else {
        endpoint = `${apiUrl}/api/v1/monitors/${id}/available-interims`;
      }
      
      const response = await fetch(endpoint);
      if (!response.ok) {
        throw new Error('Failed to fetch available interims');
      }
      const data = await response.json();
      setAvailableInterims(data);
      
      // Set default selected interim based on device type
      if (isRGDevice) {
        if (data.max_rain_gauge_interim) {
          setSelectedInterim(data.max_rain_gauge_interim);
          return data.max_rain_gauge_interim;
        }
      } else {
        if (data.max_measurement_interim) {
          setSelectedInterim(data.max_measurement_interim);
          return data.max_measurement_interim;
        }
      }
      return null;
    } catch (error) {
      console.error("Failed to fetch available interims:", error);
      return null;
    } finally {
      setLoadingInterims(false);
    }
  }, [id, deviceData]);

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
            
            // Extract the latest action from weekly quality checks
            if (data.length > 0) {
              // Find the latest check by date
              const latestCheck = data.reduce((latest, current) => {
                const latestDate = new Date(latest.check_date || 0);
                const currentDate = new Date(current.check_date || 0);
                return currentDate > latestDate ? current : latest;
              });
              
              // Extract action from the actions field
              let actionValue = 'N/A';
              if (latestCheck.actions) {
                if (typeof latestCheck.actions === 'string') {
                  actionValue = latestCheck.actions;
                } else if (typeof latestCheck.actions === 'object' && latestCheck.actions.actions) {
                  actionValue = latestCheck.actions.actions;
                }
              }
              setLatestWeeklyAction(actionValue);
            }
          }
        } catch (error) {
          setWeeklyChecksError('Failed to load data.');
        } finally {
          setLoadingWeeklyChecks(false);
        }
      }
  }, [id]);
  
  // Helper to add hours to ISO string
  const addHoursToISO = (iso, hours) => {
    return dayjs(iso).add(hours, 'hour').toISOString().slice(0, 16);
  };

  // Handle interim change
  const handleInterimChange = (newInterim) => {
    setSelectedInterim(newInterim);
    // Reset time range to default (first 3 days of the new interim)
    if (availableInterims && newInterim) {
      const apiUrl = process.env.REACT_APP_API_BASE || '';
      const isRGDevice = deviceData && deviceData.monitor_name && deviceData.monitor_name.startsWith('RG');
      
      if (isRGDevice) {
        // For RG devices, use the RG-specific endpoint
        const interimParam = `?interim=${encodeURIComponent(newInterim)}`;
        fetch(`${apiUrl}/api/v1/monitors/${id}/rg-rain-gauge${interimParam}`)
          .then(res => res.json())
          .then(rainGaugeMeta => {
            if (rainGaugeMeta.min_time) {
              setMinTime(rainGaugeMeta.min_time.slice(0, 16));
              setMaxTime(rainGaugeMeta.max_time.slice(0, 16));
              let initialStart = rainGaugeMeta.min_time.slice(0, 16);
              let initialEnd = addHoursToISO(rainGaugeMeta.min_time, 72);
              if (initialEnd > rainGaugeMeta.max_time.slice(0, 16)) initialEnd = rainGaugeMeta.max_time.slice(0, 16);
              setStartTime(initialStart);
              setEndTime(initialEnd);
              
              // Fetch data for the new interim and time range
              const rainGaugeParams = `?start=${encodeURIComponent(initialStart)}&end=${encodeURIComponent(initialEnd)}&interim=${encodeURIComponent(newInterim)}`;
              fetch(`${apiUrl}/api/v1/monitors/${id}/rg-rain-gauge${rainGaugeParams}`)
                .then(res => res.json())
                .then(data => {
                  setRainGaugeData(data.data || []);
                  setMeasurementData([]); // RG devices don't have measurement data
                });
            }
          });
      } else {
        // For regular devices, use the existing logic
        const interimParam = `?interim=${encodeURIComponent(newInterim)}`;
        fetch(`${apiUrl}/api/v1/monitors/${id}/measurements${interimParam}`)
          .then(res => res.json())
          .then(measurementMeta => {
            if (measurementMeta.min_time) {
              setMinTime(measurementMeta.min_time.slice(0, 16));
              setMaxTime(measurementMeta.max_time.slice(0, 16));
              let initialStart = measurementMeta.min_time.slice(0, 16);
              let initialEnd = addHoursToISO(measurementMeta.min_time, 72);
              if (initialEnd > measurementMeta.max_time.slice(0, 16)) initialEnd = measurementMeta.max_time.slice(0, 16);
              setStartTime(initialStart);
              setEndTime(initialEnd);
              
              // Fetch data for the new interim and time range
              const measurementParams = `?start=${encodeURIComponent(initialStart)}&end=${encodeURIComponent(initialEnd)}&interim=${encodeURIComponent(newInterim)}`;
              const rainGaugeParams = `?start=${encodeURIComponent(initialStart)}&end=${encodeURIComponent(initialEnd)}`;
              fetch(`${apiUrl}/api/v1/monitors/${id}/measurements${measurementParams}`)
                .then(res => res.json())
                .then(data => setMeasurementData(data.data || []));
              fetch(`${apiUrl}/api/v1/monitors/${id}/rain-gauge-by-time${rainGaugeParams}`)
                .then(res => res.json())
                .then(data => setRainGaugeData(data.data || []));
            }
          });
      }
    }
  };

  // Initial load: get min/max time, then fetch initial 72h data
  useEffect(() => {
    if (!id) return;
    setInitialLoading(true);
    const apiUrl = process.env.REACT_APP_API_BASE || '';
    const isRGDevice = deviceData && deviceData.monitor_name && deviceData.monitor_name.startsWith('RG');
    
    // First fetch available interims and immediately use the max interim
    fetchAvailableInterims().then((maxInterim) => {
      if (maxInterim) {
        if (isRGDevice) {
          // For RG devices, use the RG-specific endpoint
          const interimParam = `?interim=${encodeURIComponent(maxInterim)}`;
          fetch(`${apiUrl}/api/v1/monitors/${id}/rg-rain-gauge${interimParam}`)
            .then(res => res.json())
            .then(rainGaugeMeta => {
              if (!rainGaugeMeta.min_time) {
                setMeasurementData([]);
                setRainGaugeData([]);
                setStartTime('');
                setEndTime('');
                setMinTime('');
                setMaxTime('');
                setInitialLoading(false);
                return;
              }
              setMinTime(rainGaugeMeta.min_time.slice(0, 16));
              setMaxTime(rainGaugeMeta.max_time.slice(0, 16));
              let initialStart = rainGaugeMeta.min_time.slice(0, 16);
              let initialEnd = addHoursToISO(rainGaugeMeta.min_time, 72);
              if (initialEnd > rainGaugeMeta.max_time.slice(0, 16)) initialEnd = rainGaugeMeta.max_time.slice(0, 16);
              setStartTime(initialStart);
              setEndTime(initialEnd);
              // Fetch initial data for this range
              const rainGaugeParams = `?start=${encodeURIComponent(initialStart)}&end=${encodeURIComponent(initialEnd)}&interim=${encodeURIComponent(maxInterim)}`;
              fetch(`${apiUrl}/api/v1/monitors/${id}/rg-rain-gauge${rainGaugeParams}`)
                .then(res => res.json())
                .then(data => {
                  setRainGaugeData(data.data || []);
                  setMeasurementData([]); // RG devices don't have measurement data
                });
              setInitialLoading(false);
            });
        } else {
          // For regular devices, use the existing logic
          const interimParam = `?interim=${encodeURIComponent(maxInterim)}`;
          fetch(`${apiUrl}/api/v1/monitors/${id}/measurements${interimParam}`)
            .then(res => res.json())
            .then(measurementMeta => {
              if (!measurementMeta.min_time) {
                setMeasurementData([]);
                setRainGaugeData([]);
                setStartTime('');
                setEndTime('');
                setMinTime('');
                setMaxTime('');
                setInitialLoading(false);
                return;
              }
              setMinTime(measurementMeta.min_time.slice(0, 16));
              setMaxTime(measurementMeta.max_time.slice(0, 16));
              let initialStart = measurementMeta.min_time.slice(0, 16);
              let initialEnd = addHoursToISO(measurementMeta.min_time, 72);
              if (initialEnd > measurementMeta.max_time.slice(0, 16)) initialEnd = measurementMeta.max_time.slice(0, 16);
              setStartTime(initialStart);
              setEndTime(initialEnd);
              // Fetch initial data for this range
              const measurementParams = `?start=${encodeURIComponent(initialStart)}&end=${encodeURIComponent(initialEnd)}&interim=${encodeURIComponent(maxInterim)}`;
              const rainGaugeParams = `?start=${encodeURIComponent(initialStart)}&end=${encodeURIComponent(initialEnd)}`;
              fetch(`${apiUrl}/api/v1/monitors/${id}/measurements${measurementParams}`)
                .then(res => res.json())
                .then(data => setMeasurementData(data.data || []));
              fetch(`${apiUrl}/api/v1/monitors/${id}/rain-gauge-by-time${rainGaugeParams}`)
                .then(res => res.json())
                .then(data => setRainGaugeData(data.data || []));
              setInitialLoading(false);
            });
        }
      } else {
        setInitialLoading(false);
      }
    });
  }, [id, fetchAvailableInterims, deviceData]);

  // When user changes time range, fetch new data
  useEffect(() => {
    if (!id || !startTime || !endTime || initialLoading || !selectedInterim) return;
    const apiUrl = process.env.REACT_APP_API_BASE || '';
    const isRGDevice = deviceData && deviceData.monitor_name && deviceData.monitor_name.startsWith('RG');
    
    if (isRGDevice) {
      // For RG devices, use the RG-specific endpoint
      const rainGaugeParams = `?start=${encodeURIComponent(startTime)}&end=${encodeURIComponent(endTime)}&interim=${encodeURIComponent(selectedInterim)}`;
      fetch(`${apiUrl}/api/v1/monitors/${id}/rg-rain-gauge${rainGaugeParams}`)
        .then(res => res.json())
        .then(data => {
          setRainGaugeData(data.data || []);
          setMeasurementData([]); // RG devices don't have measurement data
        });
    } else {
      // For regular devices, use the existing logic
      const measurementParams = `?start=${encodeURIComponent(startTime)}&end=${encodeURIComponent(endTime)}&interim=${encodeURIComponent(selectedInterim)}`;
      const rainGaugeParams = `?start=${encodeURIComponent(startTime)}&end=${encodeURIComponent(endTime)}`;
      
      fetch(`${apiUrl}/api/v1/monitors/${id}/measurements${measurementParams}`)
        .then(res => res.json())
        .then(data => {
          setMeasurementData(data.data || []);
        });
      fetch(`${apiUrl}/api/v1/monitors/${id}/rain-gauge-by-time${rainGaugeParams}`)
        .then(res => res.json())
        .then(data => {
          setRainGaugeData(data.data || []);
        });
    }
  }, [id, startTime, endTime, initialLoading, selectedInterim, deviceData]);

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

  // Always fetch weekly checks data to get the latest action
  useEffect(() => {
    fetchWeeklyChecksData();
  }, [id, fetchWeeklyChecksData]);

  useEffect(() => {
    if (!id || !startTime || !endTime || !selectedInterim) return;
    const apiUrl = process.env.REACT_APP_API_BASE || '';
    const isRGDevice = deviceData && deviceData.monitor_name && deviceData.monitor_name.startsWith('RG');
    
    if (isRGDevice) {
      // For RG devices, use the RG-specific stats endpoint
      const params = new URLSearchParams({
        start: startTime,
        end: endTime,
        interim: selectedInterim
      });
      fetch(`${apiUrl}/api/v1/monitors/${id}/rg-stats?${params}`)
        .then(res => res.json())
        .then(data => setStatsData(data));
    } else {
      // For regular devices, use the existing stats endpoint
      const params = new URLSearchParams({
        start: startTime,
        end: endTime,
        interim: selectedInterim
      });
      fetch(`${apiUrl}/api/v1/monitors/${id}/stats?${params}`)
        .then(res => res.json())
        .then(data => setStatsData(data));
    }
  }, [id, startTime, endTime, selectedInterim, deviceData]);

  useEffect(() => {
    if (!id) return;
    const apiUrl = process.env.REACT_APP_API_BASE || '';
    const isRGDevice = deviceData && deviceData.monitor_name && deviceData.monitor_name.startsWith('RG');
    
    // Only fetch storm events for non-RG devices
    if (!isRGDevice) {
      fetch(`${apiUrl}/api/v1/monitors/${id}/storms`)
        .then(res => res.json())
        .then(data => setStormEvents(data));
    } else {
      setStormEvents([]); // RG devices don't have storm events
    }
  }, [id, deviceData]);
  
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
            comments: wrapIfString(updatedRow.comment, 'comments'),
            actions: wrapIfString(updatedRow.actions, 'actions'),
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
        removalDateValue = initialData.removal_date ? new Date(initialData.removal_date).toISOString().split('T')[0] : '';
    } else {
        removalDateValue = new Date().toISOString().split('T')[0];
    }

    setEditableResponsibility({
        action: initialData.action || '',
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
    payload.monitor_id = deviceData.id;
    payload.action = wrapIfString(payload.action, 'actions');
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
        if (!response.ok) {
          const errorText = await response.text();
          console.error('backend returned error content:', errorText);
          throw new Error("Failed to save responsibility data");
        }
        
        await fetchAllDeviceData(); // Refetch all data
        setEditMode(false);
        setEditableResponsibility(null);

    } catch (error) {
        console.error("Save failed:", error);
    }
  };

  // chart data processing
  const smartSampleData = (data, maxPoints = 200) => {
    if (data.length <= maxPoints) return data;
    
    const interval = Math.ceil(data.length / maxPoints);
    const sampled = [];
    
    for (let i = 0; i < data.length; i += interval) {
      sampled.push(data[i]);
    }
    
    // Always include the last data point
    if (sampled[sampled.length - 1] !== data[data.length - 1]) {
      sampled.push(data[data.length - 1]);
    }
    
    return sampled;
  };

  const rainfallChartData = smartSampleData(
    rainGaugeData
      .filter(item => item.intensity_mm_per_hr != null)
      .map(item => ({
        time: item.timestamp ? new Date(item.timestamp).toISOString().slice(0, 10) : '',
        timestamp: item.timestamp,
        value: Number(item.intensity_mm_per_hr)
      }))
  );
  
  const depthChartData = smartSampleData(
    measurementData
      .filter(item => item.depth != null)
      .map(item => ({
        time: item.time ? new Date(item.time).toISOString().slice(0, 10) : '',
        timestamp: item.time,
        value: Number(item.depth) / 1000 // convert mm to m
      }))
  );
    
  const flowChartData = smartSampleData(
    measurementData
      .filter(item => item.flow != null)
      .map(item => ({
        time: item.time ? new Date(item.time).toISOString().slice(0, 10) : '',
        timestamp: item.time,
        value: Number(item.flow)
      }))
  );
  
  const velocityChartData = smartSampleData(
    measurementData
      .filter(item => item.velocity != null)
      .map(item => ({
        time: item.time ? new Date(item.time).toISOString().slice(0, 10) : '',
        timestamp: item.time,
        value: Number(item.velocity)
      }))
  );

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
                      <td className="font-semibold text-gray-700 pr-2 py-1">Status</td>
                      <td className="py-1">
                        {(() => {
                          // merge status info
                          const statusInfo = [deviceData.status, deviceData.status_reason]
                            .filter(info => info && info !== '')
                            .join(' - ');
                          
                          // status color mapping
                          const getStatusColor = (status) => {
                            switch (status?.toLowerCase()) {
                              case 'active':
                                return 'bg-green-100 text-green-800';
                              case 'warning':
                                return 'bg-yellow-100 text-yellow-800';
                              case 'error':
                                return 'bg-red-100 text-red-800';
                              default:
                                return 'bg-gray-100 text-gray-800';
                            }
                          };
                          
                          return (
                            <span className={`px-2 py-1 font-medium rounded-full ${getStatusColor(deviceData.status)}`}>
                              {statusInfo || 'Unknown'}
                            </span>
                          );
                        })()}
                      </td>
                    </tr>
                    <tr>
                      <td className="font-semibold text-gray-700 pr-2 py-1">Action</td>
                      <td className="py-1 bg-yellow-100 font-semibold">
                        <input
                          type="text"
                          className={`w-full bg-yellow-100 font-semibold rounded px-2 py-1 ${editMode ? 'border border-gray-300' : 'border-none'}`}
                          value={editMode ? editableResponsibility.action : (latestWeeklyAction || 'N/A')}
                          onChange={(e) => handleResponsibilityChange('action', e.target.value)}
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
            {statsData && (
              <div className="bg-white rounded-lg shadow-md p-3 mb-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Stat Info</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm border border-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        {deviceData && deviceData.monitor_name && deviceData.monitor_name.startsWith('RG') ? (
                          <th className="px-4 py-2 border-b text-center font-normal">Rainfall</th>
                        ) : (
                          <>
                            <th className="px-4 py-2 border-b border-r text-center font-normal">Rainfall</th>
                            <th className="px-4 py-2 border-b border-r text-center font-normal">Depth</th>
                            <th className="px-4 py-2 border-b border-r text-center font-normal">Flow</th>
                            <th className="px-4 py-2 border-b text-center font-normal">Velocity</th>
                          </>
                        )}
                      </tr>
                      <tr>
                        {deviceData && deviceData.monitor_name && deviceData.monitor_name.startsWith('RG') ? (
                          <th className="px-4 py-2 border-b text-center font-normal">Peak (mm/hr) / Average (mm/hr)</th>
                        ) : (
                          <>
                            <th className="px-4 py-2 border-b border-r text-center font-normal">Peak (mm/hr) / Average (mm/hr)</th>
                            <th className="px-4 py-2 border-b border-r text-center font-normal">Min (m) / Max (m)</th>
                            <th className="px-4 py-2 border-b border-r text-center font-normal">Min (m³/s) / Max (m³/s)</th>
                            <th className="px-4 py-2 border-b text-center font-normal">Min (m/s) / Max (m/s)</th>
                          </>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        {deviceData && deviceData.monitor_name && deviceData.monitor_name.startsWith('RG') ? (
                          <td className="px-4 py-2 text-center">
                            {statsData.rainfall ? `${
                              statsData.rainfall.peak != null ? Number(statsData.rainfall.peak).toFixed(3) : '-'
                            } / ${
                              statsData.rainfall.average != null ? Number(statsData.rainfall.average).toFixed(3) : '-'
                            }` : '-'}
                          </td>
                        ) : (
                          <>
                            <td className="px-4 py-2 border-r text-center">
                              {statsData.rainfall ? `${
                                statsData.rainfall.peak != null ? Number(statsData.rainfall.peak).toFixed(3) : '-'
                              } / ${
                                statsData.rainfall.average != null ? Number(statsData.rainfall.average).toFixed(3) : '-'
                              }` : '-'}
                            </td>
                            <td className="px-4 py-2 border-r text-center">
                              {statsData.depth ? `${
                                statsData.depth.min != null ? Number(statsData.depth.min).toFixed(3) : '-'
                              } / ${
                                statsData.depth.max != null ? Number(statsData.depth.max).toFixed(3) : '-'
                              }` : '-'}
                            </td>
                            <td className="px-4 py-2 border-r text-center">
                              {statsData.flow ? `${
                                statsData.flow.min != null ? Number(statsData.flow.min).toFixed(3) : '-'
                              } / ${
                                statsData.flow.max != null ? Number(statsData.flow.max).toFixed(3) : '-'
                              }` : '-'}
                            </td>
                            <td className="px-4 py-2 text-center">
                              {statsData.velocity ? `${
                                statsData.velocity.min != null ? Number(statsData.velocity.min).toFixed(3) : '-'
                              } / ${
                                statsData.velocity.max != null ? Number(statsData.velocity.max).toFixed(3) : '-'
                              }` : '-'}
                            </td>
                          </>
                        )}
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}
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
                    {/* Interim and time range selector */}
                    <div className="flex flex-col md:flex-row md:items-center md:space-x-4 mb-4">
                      <div className="flex items-center space-x-2">
                        <label className="text-sm font-medium text-gray-700">Interim:</label>
                        <select
                          value={selectedInterim || ''}
                          onChange={e => handleInterimChange(e.target.value)}
                          className="border border-gray-300 rounded px-2 py-1"
                          disabled={loadingInterims}
                        >
                          {loadingInterims ? (
                            <option>Loading...</option>
                          ) : availableInterims ? (
                            deviceData && deviceData.monitor_name && deviceData.monitor_name.startsWith('RG') ? (
                              // For RG devices, show rain_gauge_interims
                              availableInterims.rain_gauge_interims ? (
                                availableInterims.rain_gauge_interims.map(interim => (
                                  <option key={interim} value={interim}>
                                    {interim} {interim === availableInterims.max_rain_gauge_interim ? '(Latest)' : ''}
                                  </option>
                                ))
                              ) : (
                                <option>No interims available</option>
                              )
                            ) : (
                              // For regular devices, show measurement_interims
                              availableInterims.measurement_interims ? (
                                availableInterims.measurement_interims.map(interim => (
                                  <option key={interim} value={interim}>
                                    {interim} {interim === availableInterims.max_measurement_interim ? '(Latest)' : ''}
                                  </option>
                                ))
                              ) : (
                                <option>No interims available</option>
                              )
                            )
                          ) : (
                            <option>No interims available</option>
                          )}
                        </select>
                      </div>
                      <div className="flex items-center space-x-2">
                        <label className="text-sm font-medium text-gray-700">Start Time:</label>
                        <input
                          type="datetime-local"
                          value={startTime}
                          min={minTime}
                          max={endTime}
                          onChange={e => setStartTime(e.target.value)}
                          className="border border-gray-300 rounded px-2 py-1"
                          disabled={initialLoading || !minTime}
                        />
                      </div>
                      <div className="flex items-center space-x-2">
                        <label className="text-sm font-medium text-gray-700">End Time:</label>
                        <input
                          type="datetime-local"
                          value={endTime}
                          min={startTime}
                          max={maxTime}
                          onChange={e => setEndTime(e.target.value)}
                          className="border border-gray-300 rounded px-2 py-1"
                          disabled={initialLoading || !maxTime}
                        />
                      </div>
                    </div>
                    {initialLoading ? (
                      <div className="text-center text-gray-500 py-8">Loading data...</div>
                    ) : (
                      <>
                        {/* For RG devices, only show rainfall intensity chart */}
                        {deviceData && deviceData.monitor_name && deviceData.monitor_name.startsWith('RG') ? (
                          <div>
                            <h3 className="text-lg font-semibold text-gray-900 mb-2">Rainfall intensity (mm/hr)</h3>
                            <ResponsiveContainer width="100%" height={180}>
                              <LineChart data={rainfallChartData} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
                                <CartesianGrid stroke="#e0e0e0" strokeDasharray="0" />
                                <XAxis 
                                  dataKey="time" 
                                  stroke="#6b7280" 
                                  tick={{ fontSize: 12 }}
                                  interval="preserveStartEnd"
                                />
                                <YAxis domain={['dataMin', 'dataMax']} tick={{ fontSize: 12 }} />
                                <Tooltip content={<CustomTooltip title="Rainfall Intensity" unit="mm/hr" />} />
                                <Line
                                  type="monotone"
                                  dataKey="value"
                                  stroke="#2563eb"
                                  strokeWidth={1}
                                  dot={false}
                                  isAnimationActive={false}
                                />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        ) : (
                          <>
                            {/* Depth chart */}
                            <div>
                              <h3 className="text-lg font-semibold text-gray-900 mb-2">Depth (m)</h3>
                              <ResponsiveContainer width="100%" height={180}>
                                <LineChart data={depthChartData} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
                                  <CartesianGrid stroke="#e0e0e0" strokeDasharray="0" />
                                  <XAxis 
                                    dataKey="time" 
                                    stroke="#6b7280" 
                                    tick={{ fontSize: 12 }}
                                    interval="preserveStartEnd"
                                  />
                                  <YAxis domain={['dataMin', 'dataMax']} tick={{ fontSize: 12 }} />
                                  <Tooltip content={<CustomTooltip title="Depth" unit="m" />} />
                                  <Line
                                    type="monotone"
                                    dataKey="value"
                                    stroke="#ef4444"
                                    strokeWidth={1}
                                    dot={false}
                                    isAnimationActive={false}
                                  />
                                </LineChart>
                              </ResponsiveContainer>
                            </div>
                            {/* Flow chart */}
                            <div>
                              <h3 className="text-lg font-semibold text-gray-900 mb-2">Flow (m3/s)</h3>
                              <ResponsiveContainer width="100%" height={180}>
                                <LineChart data={flowChartData} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
                                  <CartesianGrid stroke="#e0e0e0" strokeDasharray="0" />
                                  <XAxis 
                                    dataKey="time" 
                                    stroke="#6b7280" 
                                    tick={{ fontSize: 12 }}
                                    interval="preserveStartEnd"
                                  />
                                  <YAxis domain={['dataMin', 'dataMax']} tick={{ fontSize: 12 }} />
                                  <Tooltip content={<CustomTooltip title="Flow" unit="m³/s" />} />
                                  <Line
                                    type="monotone"
                                    dataKey="value"
                                    stroke="#ef4444"
                                    strokeWidth={1}
                                    dot={false}
                                    isAnimationActive={false}
                                  />
                                </LineChart>
                              </ResponsiveContainer>
                            </div>
                            {/* Velocity chart */}
                            <div>
                              <h3 className="text-lg font-semibold text-gray-900 mb-2">Velocity (m/s)</h3>
                              <ResponsiveContainer width="100%" height={180}>
                                <LineChart data={velocityChartData} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
                                  <CartesianGrid stroke="#e0e0e0" strokeDasharray="0" />
                                  <XAxis 
                                    dataKey="time" 
                                    stroke="#6b7280" 
                                    tick={{ fontSize: 12 }}
                                    interval="preserveStartEnd"
                                  />
                                  <YAxis domain={['dataMin', 'dataMax']} tick={{ fontSize: 12 }} />
                                  <Tooltip content={<CustomTooltip title="Velocity" unit="m/s" />} />
                                  <Line
                                    type="monotone"
                                    dataKey="value"
                                    stroke="#ef4444"
                                    strokeWidth={1}
                                    dot={false}
                                    isAnimationActive={false}
                                  />
                                </LineChart>
                              </ResponsiveContainer>
                            </div>
                            {/* Rainfall intensity chart (at the bottom) */}
                            <div>
                              <h3 className="text-lg font-semibold text-gray-900 mb-2">Rainfall intensity (mm/hr)</h3>
                              <ResponsiveContainer width="100%" height={180}>
                                <LineChart data={rainfallChartData} margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
                                  <CartesianGrid stroke="#e0e0e0" strokeDasharray="0" />
                                  <XAxis 
                                    dataKey="time" 
                                    stroke="#6b7280" 
                                    tick={{ fontSize: 12 }}
                                    interval="preserveStartEnd"
                                  />
                                  <YAxis domain={['dataMin', 'dataMax']} tick={{ fontSize: 12 }} />
                                  <Tooltip content={<CustomTooltip title="Rainfall Intensity" unit="mm/hr" />} />
                                  <Line
                                    type="monotone"
                                    dataKey="value"
                                    stroke="#2563eb"
                                    strokeWidth={1}
                                    dot={false}
                                    isAnimationActive={false}
                                  />
                                </LineChart>
                              </ResponsiveContainer>
                            </div>
                          </>
                        )}
                      </>
                    )}
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
            {/* Storm Events Cards - Only show for non-RG devices */}
            {deviceData && deviceData.monitor_name && deviceData.monitor_name.startsWith('RG') ? (
              <div className="bg-white rounded-lg shadow-md p-4 text-gray-500">Rain gauge devices do not show storm events.</div>
            ) : (
              <>
                {stormEvents.length === 0 ? (
                  <div className="bg-white rounded-lg shadow-md p-4 text-gray-500">No storm events found.</div>
                ) : (
                  stormEvents.map((event, idx) => (
                    <div key={event.id} className="bg-white rounded-lg shadow-md p-4">
                      <h4 className="font-semibold text-gray-900 mb-3">
                        Storm {event.storm_type || String.fromCharCode(65 + idx)}
                      </h4>
                      <div className="space-y-3">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Date</span>
                          <span className="text-sm font-medium">
                            {event.start_time ? new Date(event.start_time).toLocaleDateString() : ''}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Total Rain (mm)</span>
                          <span className="text-sm font-medium">
                            {event.event_comment?.total_rain != null ? Number(event.event_comment.total_rain).toFixed(3) : '-'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Peak intensity (mm/hr)</span>
                          <span className="text-sm font-medium">
                            {event.event_comment?.peak_intensity != null ? Number(event.event_comment.peak_intensity).toFixed(3) : '-'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-600">Average duration (min)</span>
                          <span className="text-sm font-medium">
                            {event.event_comment?.duration_minutes != null ? Number(event.event_comment.duration_minutes).toFixed(0) : '-'}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </>
            )}
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
    setEditRows(rws => rws.map((r, i) =>
      i === idx
        ? { ...r, [field]: field === 'actions' ? wrapIfString(value, 'actions') : value }
        : r
    ));
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