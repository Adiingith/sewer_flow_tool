import React, { useState, useMemo, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Search, 
  ChevronLeft,
  ChevronRight,
  ClipboardCheck,
  CalendarCheck,
  ClipboardEdit
} from 'lucide-react';
import PresiteInstallCheckModal from './PresiteInstallCheckModal';
import WeeklyQualityCheckModal from './WeeklyQualityCheckModal';
import ActionResponsibilityModal from './ActionResponsibilityModal';

function Pagination({ currentPage, totalPages, onPageChange }) {
  if (totalPages <= 1) return null;

  return (
    <div className="flex items-center justify-between mt-4">
      <button
        onClick={() => onPageChange(currentPage - 1)}
        disabled={currentPage === 1}
        className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <ChevronLeft className="w-5 h-5" />
      </button>
      <span className="text-sm text-gray-700">
        Page {currentPage} of {totalPages}
      </span>
      <button
        onClick={() => onPageChange(currentPage + 1)}
        disabled={currentPage === totalPages}
        className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <ChevronRight className="w-5 h-5" />
      </button>
    </div>
  );
}

function DeviceTable({ 
  devices, 
  loading = false, 
  currentPage, 
  totalPages, 
  onPageChange,
  selectedDevices,
  onSelectionChange
}) {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [isPresiteModalOpen, setIsPresiteModalOpen] = useState(false);
  const [isWeeklyModalOpen, setIsWeeklyModalOpen] = useState(false);
  const [monitorsForModal, setMonitorsForModal] = useState([]);
  const [isActionModalOpen, setIsActionModalOpen] = useState(false);
  const navigate = useNavigate();
  const masterCheckboxRef = useRef();

  const filteredDevices = useMemo(() => {
    return devices.filter(device => {
      const searchTermLower = searchTerm.toLowerCase();
      const matchesSearch = searchTerm === '' || 
        device.name.toLowerCase().includes(searchTermLower) ||
        device.location.toLowerCase().includes(searchTermLower) ||
        device.mhReference.toLowerCase().includes(searchTermLower);
      
      const matchesStatus = statusFilter === 'all' || device.status === statusFilter;

      return matchesSearch && matchesStatus;
    });
  }, [devices, searchTerm, statusFilter]);

  useEffect(() => {
    if (masterCheckboxRef.current) {
      const nonScrappedOnPage = filteredDevices.filter(d => d.status !== 'scrapped');
      const numSelectedOnPage = nonScrappedOnPage.filter(d => selectedDevices.includes(d.id)).length;
      
      const allSelected = numSelectedOnPage > 0 && numSelectedOnPage === nonScrappedOnPage.length;
      const someSelected = numSelectedOnPage > 0 && numSelectedOnPage < nonScrappedOnPage.length;

      masterCheckboxRef.current.checked = allSelected;
      masterCheckboxRef.current.indeterminate = someSelected;
    }
  }, [selectedDevices, filteredDevices]);

  const handleSelectAll = async (e) => {
    if (e.target.checked) {
      try {
        const apiUrl = process.env.REACT_APP_API_BASE || '';
        const response = await fetch(`${apiUrl}/api/v1/monitors/ids`);
        if (!response.ok) throw new Error('Failed to fetch all device IDs');
        const ids = await response.json();
        onSelectionChange(ids);
      } catch (error) {
        console.error("Error in handleSelectAll:", error);
        // Deselect checkbox if API call fails
        if (masterCheckboxRef.current) {
            masterCheckboxRef.current.checked = false;
        }
        onSelectionChange([]);
      }
    } else {
      onSelectionChange([]);
    }
  };

  const handleSelectOne = (e, id) => {
    if (e.target.checked) {
      onSelectionChange([...selectedDevices, id]);
    } else {
      onSelectionChange(selectedDevices.filter(deviceId => deviceId !== id));
    }
  };

  const fetchMonitorsAndOpenModal = async (modalType) => {
    if (selectedDevices.length === 0) return;

    try {
      const apiUrl = process.env.REACT_APP_API_BASE || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/v1/monitors/by_ids`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(selectedDevices)
      });
      if (!response.ok) throw new Error('Failed to fetch monitor details');
      
      const selectedMonitors = await response.json();
      setMonitorsForModal(selectedMonitors);
      
      if (modalType === 'presite') {
        setIsPresiteModalOpen(true);
      } else if (modalType === 'weekly') {
        setIsWeeklyModalOpen(true);
      } else if (modalType === 'action') {
        setIsActionModalOpen(true);
      }
    } catch (error) {
      console.error("Failed to fetch monitor details for modal:", error);
      // Optionally show an error toast/notification to the user
    }
  };

  const handleOpenPresiteCheck = () => {
    fetchMonitorsAndOpenModal('presite');
  };

  const handleOpenWeeklyCheck = () => {
    fetchMonitorsAndOpenModal('weekly');
  };

  const handleRowClick = (deviceId) => {
    navigate(`/devices/${deviceId}`);
  };

  const handleOpenActionModal = () => {
    fetchMonitorsAndOpenModal('action');
  };

  const handleSaveActions = async ({ toUpdate, toCreate }) => {
    const apiUrl = process.env.REACT_APP_API_BASE || '';
    const promises = [];

    if (toUpdate && toUpdate.length > 0) {
        promises.push(fetch(`${apiUrl}/api/v1/responsibilities/bulk_update`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(toUpdate)
        }));
    }

    if (toCreate && toCreate.length > 0) {
        promises.push(fetch(`${apiUrl}/api/v1/responsibilities/bulk_create`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(toCreate)
        }));
    }
    
    if (promises.length === 0) return;

    const responses = await Promise.all(promises);
    for (const response of responses) {
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'An unknown error occurred' }));
            console.error("Backend validation error:", errorData.detail);

            let errorMessage = 'An unknown error occurred.';
            if (errorData.detail && Array.isArray(errorData.detail)) {
                errorMessage = errorData.detail.map(e => 
                    `Field '${e.loc.join('.')}' - ${e.msg}`
                ).join('; ');
            } else if (errorData.detail) {
                errorMessage = JSON.stringify(errorData.detail);
            }

            throw new Error(`Failed to save: ${errorMessage}`);
        }
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-12 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md">
      {/* Table header controls */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-2 sm:space-y-0">
          <div className="flex items-center space-x-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type="text"
                placeholder="Search devices..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Status</option>
              <option value="active">Active</option>
              <option value="warning">Warning</option>
              <option value="error">Error</option>
            </select>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={handleOpenActionModal}
              disabled={selectedDevices.length === 0}
              className="px-4 py-2 text-sm font-medium rounded-md flex items-center space-x-2 text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed"
            >
              <ClipboardEdit className="w-4 h-4" />
              <span>Action Manage</span>
            </button>
            <button
              onClick={handleOpenPresiteCheck}
              disabled={selectedDevices.length === 0}
              className="px-4 py-2 text-sm font-medium rounded-md flex items-center space-x-2 text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed"
            >
              <ClipboardCheck className="w-4 h-4" />
              <span>Presite & Install Check</span>
            </button>
            <button
              onClick={handleOpenWeeklyCheck}
              disabled={selectedDevices.length === 0}
              className="px-4 py-2 text-sm font-medium rounded-md flex items-center space-x-2 text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed"
            >
              <CalendarCheck className="w-4 h-4" />
              <span>Weekly Quality Check</span>
            </button>
          </div>
        </div>
      </div>

      {/* Table content */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 table-fixed">
          <thead className="bg-gray-50">
            <tr>
              <th scope="col" className="px-6 py-3 w-12">
                <input
                  type="checkbox"
                  ref={masterCheckboxRef}
                  className="rounded border-gray-300"
                  onChange={handleSelectAll}
                />
              </th>
              <th scope="col" className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-20">
                FM / DM / PL
              </th>
              <th scope="col" className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-24">
                Install Date
              </th>
              <th scope="col" className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-20">
                W3W
              </th>
              <th scope="col" className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Location
              </th>
              <th scope="col" className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-28">
                MH Reference
              </th>
              <th scope="col" className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-16">
                Pipe
              </th>
              <th scope="col" className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-16">
                Shape
              </th>
              <th scope="col" className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-28">
                Dimensions (mm)
              </th>
              <th scope="col" className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-24">
                Status
              </th>
              <th scope="col" className="px-3 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-32">
                Action
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200 text-xs">
            {filteredDevices.map((device) => {
              if (device.status === 'scrapped') {
                return (
                  <tr key={device.id} className="bg-gray-100 text-gray-500">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <input
                        type="checkbox"
                        disabled={true}
                        className="rounded border-gray-300"
                      />
                    </td>
                    <td className="px-3 py-3 font-medium">{device.name}</td>
                    <td className="px-3 py-3 italic">scrapped</td>
                    <td className="px-3 py-3 italic">scrapped</td>
                    <td className="px-3 py-3 italic">scrapped</td>
                    <td className="px-3 py-3 italic">scrapped</td>
                    <td className="px-3 py-3 italic">scrapped</td>
                    <td className="px-3 py-3 italic">scrapped</td>
                    <td className="px-3 py-3 italic">scrapped</td>
                    <td className="px-3 py-3 italic">scrapped</td>
                    <td className="px-3 py-3 italic">
                      <div className="truncate">scrapped</div>
                    </td>
                  </tr>
                );
              }

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

              // merge dimensions
              const dimensions = [device.height, device.width, device.depth]
                .filter(dim => dim && dim !== '')
                .join(' - ');

              return (
                <tr
                  key={device.id}
                  onClick={() => handleRowClick(device.id)}
                  className="hover:bg-gray-50 cursor-pointer"
                >
                  <td className="px-6 py-4 whitespace-nowrap">
                    <input
                      type="checkbox"
                      onChange={(e) => handleSelectOne(e, device.id)}
                      checked={selectedDevices.includes(device.id)}
                      onClick={(e) => e.stopPropagation()}
                      disabled={device.status === 'scrapped'}
                      className="rounded border-gray-300"
                    />
                  </td>
                  <td className="px-3 py-3">{device.name}</td>
                  <td className="px-3 py-3">{device.installDate ? device.installDate.slice(0, 10) : ''}</td>
                  <td className="px-3 py-3">{device.w3w}</td>
                  <td className="px-3 py-3">{device.location}</td>
                  <td className="px-3 py-3">{device.mhReference}</td>
                  <td className="px-3 py-3">{device.pipe}</td>
                  <td className="px-3 py-3">{device.shape}</td>
                  <td className="px-3 py-3" title={`Height: ${device.height || '-'} | Width: ${device.width || '-'} | Depth: ${device.depth || '-'}`}>
                    {dimensions || '-'}
                  </td>
                  <td className="px-3 py-3">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(device.status)}`}>
                      {device.status || 'Unknown'}
                    </span>
                  </td>
                  <td className="px-3 py-3 bg-yellow-100 font-semibold">
                    <div 
                      className="truncate cursor-help" 
                      title={device.action || ''}
                    >
                      {device.action}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {filteredDevices.length === 0 && (
        <div className="text-center py-12">
          <div className="text-gray-400 text-lg mb-2">No matching devices found</div>
          <div className="text-gray-500 text-sm">Try adjusting your search or filters</div>
        </div>
      )}
      
      <div className="px-4 py-3 border-t border-gray-200">
        <Pagination currentPage={currentPage} totalPages={totalPages} onPageChange={onPageChange} />
      </div>

      {isPresiteModalOpen && (
        <PresiteInstallCheckModal 
          monitors={monitorsForModal} 
          onClose={() => setIsPresiteModalOpen(false)} 
        />
      )}
      {isWeeklyModalOpen && (
        <WeeklyQualityCheckModal 
          monitors={monitorsForModal}
          onClose={() => setIsWeeklyModalOpen(false)}
        />
      )}
      <ActionResponsibilityModal
        isOpen={isActionModalOpen}
        onClose={() => setIsActionModalOpen(false)}
        selectedMonitors={monitorsForModal}
        onSave={handleSaveActions}
      />
    </div>
  );
}

export default DeviceTable; 