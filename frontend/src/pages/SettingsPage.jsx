import React, { useEffect, useState } from 'react';

const API_BASE = process.env.REACT_APP_API_BASE;

function fetcher(url, options) {
  return fetch(url, options).then(res => {
    if (!res.ok) throw new Error('Network error');
    return res.json();
  });
}

export default function SettingsPage() {
  // Rain gauge assignment state
  const [areas, setAreas] = useState([]);
  const [selectedArea, setSelectedArea] = useState('');
  const [monitors, setMonitors] = useState([]); // Flow Survey Sheet
  const [rainGauges, setRainGauges] = useState([]); // Rainfall Assessment Sheet
  const [selectedRG, setSelectedRG] = useState({}); // {monitor_id: rg_monitor_id}
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  // Fetch area list on mount
  useEffect(() => {
    fetcher(`${API_BASE}/api/v1/monitor-analysis/areas`).then(data => {
      setAreas(data.areas || []);
      if (data.areas && data.areas.length > 0 && !selectedArea) {
        setSelectedArea(data.areas[0]);
      }
    });
  }, []);

  // Fetch monitors and rain gauges when area/message变化
  useEffect(() => {
    if (!selectedArea) return;
    fetcher(`${API_BASE}/api/v1/monitor-analysis/monitors/with-rain-gauge?area=${encodeURIComponent(selectedArea)}`)
      .then(data => setMonitors(data.monitors || []));
    fetcher(`${API_BASE}/api/v1/monitor-analysis/monitors/rain-gauges?area=${encodeURIComponent(selectedArea)}`)
      .then(data => setRainGauges(data.rain_gauges || []));
  }, [selectedArea, message]);

  // Assign rain gauge for a specific monitor
  const handleAssign = async (monitor) => {
    const rgId = selectedRG[monitor.monitor_id];
    if (!rgId) return;
    setLoading(true);
    setMessage('');
    try {
      await fetcher(`${API_BASE}/api/v1/monitor-analysis/monitor/${monitor.monitor_id}/assign-rain-gauge?monitor_id=${monitor.monitor_id}&interim=${rgId}:default`, { method: 'POST' });
      setMessage('assigned successfully');
    } catch (e) {
      setMessage('assignment failed: ' + e.message);
    }
    setLoading(false);
  };

  // Unassign rain gauge
  const handleUnassign = async (monitorId) => {
    setLoading(true);
    setMessage('');
    try {
      await fetcher(`${API_BASE}/api/v1/monitor-analysis/monitor/${monitorId}/unassign-rain-gauge`, { method: 'DELETE' });
      setMessage('unassigned successfully');
    } catch (e) {
      setMessage('unassignment failed: ' + e.message);
    }
    setLoading(false);
  };

  return (
    <div className="max-w-4xl mx-auto py-8 px-4">
      <h1 className="text-2xl font-bold mb-6">system settings</h1>
      <section className="mb-10">
        <h2 className="text-xl font-semibold mb-4">rain gauge assignment management</h2>
        <div className="mb-6 flex items-center gap-4">
          <label className="font-medium">select project (area):</label>
          <select
            className="border rounded px-2 py-1 w-64"
            value={selectedArea}
            onChange={e => setSelectedArea(e.target.value)}
          >
            {areas.map(area => (
              <option key={area} value={area}>{area}</option>
            ))}
          </select>
        </div>
        <div className="bg-white rounded shadow p-4">
          <h3 className="font-semibold mb-2">flow monitoring devices</h3>
          <table className="w-full text-sm border">
            <thead>
              <tr className="bg-gray-100">
                <th className="p-2 border">flow monitoring device</th>
                <th className="p-2 border">assigned rain gauge</th>
                <th className="p-2 border">select rain gauge</th>
                <th className="p-2 border">operation</th>
              </tr>
            </thead>
            <tbody>
              {monitors.length === 0 && (
                <tr><td colSpan={4} className="text-center p-4">no device</td></tr>
              )}
              {monitors.map(m => (
                <tr key={m.monitor_id}>
                  <td className="border p-2">{m.monitor_name.replace(/\s*\(.*\)/, '')}</td>
                  <td className="border p-2">{
                    (() => {
                      const rg = rainGauges.find(rg => rg.id === m.assigned_rain_gauge_id);
                      return rg ? rg.monitor_name.replace(/\s*\(.*\)/, '') : '-';
                    })()
                  }</td>
                  <td className="border p-2">
                    <select
                      className="border rounded px-2 py-1 w-48"
                      value={selectedRG[m.monitor_id] || ''}
                      onChange={e => setSelectedRG({ ...selectedRG, [m.monitor_id]: e.target.value })}
                    >
                      <option value="">please select</option>
                      {rainGauges.map(rg => (
                        <option key={rg.monitor_id} value={rg.monitor_id}>{rg.monitor_name.replace(/\s*\(.*\)/, '')}</option>
                      ))}
                    </select>
                  </td>
                  <td className="border p-2">
                    <button
                      className="bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 disabled:opacity-50 mr-2"
                      onClick={() => handleAssign(m)}
                      disabled={loading || !selectedRG[m.monitor_id]}
                    >
                      assign
                    </button>
                    <button
                      className="text-red-600 hover:underline"
                      onClick={() => handleUnassign(m.monitor_id)}
                      disabled={loading}
                    >
                      unassign
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {message && <div className="mt-3 text-blue-600">{message}</div>}
        </div>
      </section>
    </div>
  );
} 