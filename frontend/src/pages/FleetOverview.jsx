import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Trash2, 
  Archive,
  PieChart as PieChartIcon,
  BarChart2,
  Calendar,
  Download,
  Filter,
  Users
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend
} from 'recharts';
import KPICard from '../components/KPICard';
import DeviceTable from '../components/DeviceTable';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#AF19FF'];

function FleetOverview() {
  const [loading, setLoading] = useState(true);
  const [devices, setDevices] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [selectedDevices, setSelectedDevices] = useState([]);
  const [itemsPerPage, setItemsPerPage] = useState(8);
  // const itemsPerPage = 8;

  // State for the new dashboard data
  const [summaryData, setSummaryData] = useState(null);
  const [dailyRemovals, setDailyRemovals] = useState([]);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        const apiUrl = process.env.REACT_APP_API_BASE || '';
        
        // Fetch summary data
        const summaryResponse = await fetch(`${apiUrl}/api/v1/visualization/dashboard_summary`);
        if (!summaryResponse.ok) throw new Error('Failed to fetch summary data');
        const summary = await summaryResponse.json();
        setSummaryData(summary);

        // Fetch daily removals data
        const removalsResponse = await fetch(`${apiUrl}/api/v1/visualization/daily_removals`);
        if (!removalsResponse.ok) throw new Error('Failed to fetch daily removals data');
        const removals = await removalsResponse.json();
        setDailyRemovals(removals);

      } catch (error) {
        console.error("Error fetching dashboard data:", error);
      }
    };

    const fetchDevices = async (page) => {
        // This part remains unchanged, but you might want to integrate its loading state
        // with the new dashboard loading logic. For now, we keep it separate.
        try {
            const apiUrl = process.env.REACT_APP_API_BASE || '';
            const response = await fetch(`${apiUrl}/api/v1/monitors?page=${page}&limit=${itemsPerPage}`);
            if (!response.ok) throw new Error('Failed to fetch devices');
            const data = await response.json();
            const mappedData = data.data.map(d => ({
                id: d.id,
                name: d.monitor_name,
                status: d.status,
                installDate: d.install_date,
                w3w: d.w3w,
                location: d.location,
                mhReference: d.mh_reference,
                pipe: d.pipe,
                height: d.height_mm,
                width: d.width_mm,
                shape: d.shape,
                depth: d.depth_mm,
                action: d.action || '',
            }));
            setDevices(mappedData);
            setTotalPages(Math.ceil(data.total / data.limit));
        } catch (error) {
            console.error("Error fetching devices:", error);
        }
    };

    Promise.all([fetchDashboardData(), fetchDevices(currentPage)]).finally(() => setLoading(false));

  }, [currentPage, itemsPerPage]);

  const handlePageChange = (newPage) => {
    if (newPage > 0 && newPage <= totalPages) {
      setCurrentPage(newPage);
    }
  };

    const handleItemsPerPageChange = (newItemsPerPage) => {
        if (newItemsPerPage > 0 && newItemsPerPage <= 100) {
            setItemsPerPage(newItemsPerPage);
            setCurrentPage(1); // Reset to first page when items per page changes
        }
    };

  const handleExportCSV = () => {
  };

  const handleSelectionChange = (newSelection) => {
    setSelectedDevices(newSelection);
  };
  
  const categoryDistributionData = summaryData ? 
    Object.entries(summaryData.category_counts).map(([name, value]) => ({ name, value }))
    : [];

  if (loading) {
    return <div>Loading...</div>; // Simple loading state
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Page header */}
      <div className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between items-center py-4">
                  <div>
                      <h1 className="text-2xl font-bold text-gray-900 mt-1">Fleet Overview</h1>
                  </div>
                  <div className="flex items-center space-x-3">
                      <button
                          onClick={handleExportCSV}
                          className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                      >
                          <Download className="w-4 h-4" />
                          <span>Export Report</span>
                      </button>
                  </div>
              </div>
          </div>
      </div>


      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* KPI cards area */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <KPICard
            title={<><span>FM</span><br/><span className="text-xs text-gray-500">Removals / Total</span></>}
            value={
              summaryData && summaryData.category_removal_counts && summaryData.category_counts
                ? `${summaryData.category_removal_counts.FM || 0} / ${summaryData.category_counts.FM || 0}`
                : '0 / 0'
            }
            color="blue"
          />
          <KPICard
            title={<><span>DM</span><br/><span className="text-xs text-gray-500">Removals / Total</span></>}
            value={
              summaryData && summaryData.category_removal_counts && summaryData.category_counts
                ? `${summaryData.category_removal_counts.DM || 0} / ${summaryData.category_counts.DM || 0}`
                : '0 / 0'
            }
            color="green"
          />
          <KPICard
            title={<><span>PL</span><br/><span className="text-xs text-gray-500">Removals / Total</span></>}
            value={
              summaryData && summaryData.category_removal_counts && summaryData.category_counts
                ? `${summaryData.category_removal_counts.PL || 0} / ${summaryData.category_counts.PL || 0}`
                : '0 / 0'
            }
            color="orange"
          />
          <KPICard
            title={<><span>RG</span><br/><span className="text-xs text-gray-500">Removals / Total</span></>}
            value={
              summaryData && summaryData.category_removal_counts && summaryData.category_counts
                ? `${summaryData.category_removal_counts.RG || 0} / ${summaryData.category_counts.RG || 0}`
                : '0 / 0'
            }
            color="red"
          />
        </div>

        {/* Charts area */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Daily Removals Chart */}
          <div className="lg:col-span-2 bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Device Removals by Date</h3>
              <p className="text-sm text-gray-500 mb-2">Only dates with device removals are shown</p>
              <BarChart2 className="w-5 h-5 text-gray-400" />
            </div>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={dailyRemovals}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="date" stroke="#6b7280" tick={{ fontSize: 12 }} />
                <YAxis stroke="#6b7280" />
                <Tooltip
                  labelStyle={{ color: '#374151' }}
                  contentStyle={{ borderRadius: '0.5rem',
                  boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)'}}
                />
                <Bar dataKey="count" fill="#22C55E" name="Removals" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Device Category Distribution */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Device Category Distribution</h3>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={categoryDistributionData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  nameKey="name"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {categoryDistributionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        {/* Device table area remains, but we remove the selection logic for now to simplify */}
        <div className="bg-white rounded-lg shadow-md">
            <div className="p-6">
                <h3 className="text-lg font-semibold text-gray-900">Device List</h3>
            </div>
            <DeviceTable 
                devices={devices} 
                onSelectionChange={handleSelectionChange} 
                selectedDevices={selectedDevices}
                currentPage={currentPage}
                totalPages={totalPages}
                itemsPerPage={itemsPerPage}
                onPageChange={handlePageChange}
                handleItemsPerPageChange={handleItemsPerPageChange}
            />
        </div>
      </div>
    </div>
  );
}

export default FleetOverview; 