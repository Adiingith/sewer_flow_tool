import React, { useRef, useState } from 'react';
import { Search, Plus, X, Upload } from 'lucide-react';
import { Link } from 'react-router-dom';

const API_BASE = process.env.REACT_APP_API_BASE || '';

const FILE_TYPES = [
    { value: "Flow Survey Sheet", label: "Flow Survey Sheet" },
    { value: "Rainfall Assessment Sheet", label: "Rainfall Assessment Sheet" },
    { value: "Monitor Time Series Data", label: "Monitor Time Series Data" },
];

function HomePage() {
  const fileInputRef = useRef(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [fileList, setFileList] = useState([]);
  const [isBatchUploading, setIsBatchUploading] = useState(false);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });
  const [isConfirmModalOpen, setIsConfirmModalOpen] = useState(false);
  const [batchType, setBatchType] = useState(FILE_TYPES[0].value);
  const [batchArea, setBatchArea] = useState("");
  const [batchInterim, setBatchInterim] = useState("");

  const handleUploadClick = () => {
    setIsModalOpen(true);
  };

  const resetModalState = () => {
    setIsModalOpen(false);
    setFileList([]);
  };

  const handleChooseFiles = () => {
    setBatchType(FILE_TYPES[0].value);
    setBatchArea("");
    setBatchInterim("");
    setIsConfirmModalOpen(true);
  };

  const handleFileChange = async (event) => {
    const selectedFiles = Array.from(event.target.files);
    if (fileInputRef.current) {
        fileInputRef.current.value = "";
    }
    if (!selectedFiles.length) return;
    setFileList(prev => [
      ...prev,
      ...selectedFiles.map(file => ({
        id: `${file.name}-${file.lastModified}`,
        file,
        name: file.name,
        status: 'pending',
        message: 'Ready to upload',
        type: batchType,
        area: batchArea,
        interim: batchType === 'Monitor Time Series Data' ? batchInterim : '',
      }))
    ]);
  };

  const handleMetadataChange = (id, field, value) => {
    setFileList(prev => 
        prev.map(f => (f.id === id ? { ...f, [field]: value } : f))
    );
  };

  const handleRemoveFile = (id) => {
    setFileList(prev => prev.filter(f => f.id !== id));
  };

  const handleBatchUpload = async () => {
    const pendingFiles = fileList.filter(f => f.status === 'pending');
    if (pendingFiles.length === 0) return;
    setIsBatchUploading(true);
    setBatchProgress({ current: 0, total: pendingFiles.length });
    for (let i = 0; i < pendingFiles.length; i++) {
      await handleUploadFile(pendingFiles[i].id, true);
      setBatchProgress({ current: i + 1, total: pendingFiles.length });
    }
    setIsBatchUploading(false);
  };

  const handleUploadFile = async (id, silent = false) => {
    const fileToUpload = fileList.find(f => f.id === id);
    if (!fileToUpload || !fileToUpload.type || !fileToUpload.area) {
      if (!silent) alert("Please select a Type and enter an Area for the file.");
      return;
    }
    if (fileToUpload.type === "Monitor Time Series Data") {
      if (!fileToUpload.interim || !/^\d+$/.test(fileToUpload.interim)) {
        if (!silent) alert("Please enter a valid interim number! (e.g., 1, 2, 3, 4)");
        return;
      }
    }
    setFileList(prev => 
      prev.map(f => (f.id === id ? { ...f, status: 'uploading', message: 'Uploading...' } : f))
    );
    const formData = new FormData();
    formData.append("file", fileToUpload.file);
    formData.append("model_type", fileToUpload.type);
    formData.append("area", fileToUpload.area);
    if (fileToUpload.type === "Monitor Time Series Data") {
      formData.append("interim", fileToUpload.interim);
    }
    try {
      const response = await fetch(`${API_BASE}/api/v1/upload_file`, {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      if (response.ok) {
        setFileList(prev => 
          prev.map(f => (f.id === id ? { ...f, status: 'success', message: result.message } : f))
        );
      } else {
        setFileList(prev => 
          prev.map(f => (f.id === id ? { ...f, status: 'error', message: result.detail || 'Upload failed' } : f))
        );
      }
    } catch (error) {
      console.error("Upload error:", error);
      setFileList(prev => 
        prev.map(f => (f.id === id ? { ...f, status: 'error', message: 'Network error or server unavailable.' } : f))
      );
    }
  };


  return (
    <div className="relative min-h-screen bg-white">
      {/* Search bar and navigation */}
      <div className="absolute top-1/2 left-0 w-full transform -translate-y-1/2 flex justify-center">
        <div className="w-full max-w-xl flex items-center shadow-lg rounded-lg px-6 py-4 border border-gray-200 bg-white text-lg">
          <button className="mr-2 text-gray-600 hover:text-gray-800">
            <Search size={20} />
          </button>
          <input
            type="text"
            className="flex-1 bg-transparent outline-none text-gray-700 placeholder-gray-400 px-3"
            placeholder="Search devices or files..."
          />
          <button 
            className="ml-2 text-gray-600 hover:text-gray-800" 
            onClick={handleUploadClick}
          >
            <Plus size={20} />
          </button>
          <Link 
            to="/fleet"
            className="ml-4 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            View Fleet
          </Link>
        </div>
      </div>

      {/* Logo and title */}
      <div className="absolute left-1/2 top-[45%] transform -translate-x-1/2 -translate-y-full flex flex-col items-center">
        <div className="mb-4">
          <div className="w-24 h-24 bg-white border border-gray-200 rounded-lg flex items-center justify-center shadow-md">
            <img src="/ARUP_LOGO.png" alt="ARUP Logo" className="object-contain w-full h-full" />
          </div>
        </div>
        <h1 className="text-3xl font-bold text-gray-800">Sewer Flow Monitoring System</h1>
        <p className="text-gray-600 mt-2">Real-time Water Quality Assessment</p>
      </div>

      {/* Upload modal */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-30 flex items-center justify-center z-50">
          <div className="relative bg-white p-6 rounded-lg shadow-xl w-[900px] max-h-[85vh] flex flex-col">
            {/* Top control area */}
            <div className="absolute top-4 right-4">
              <button
                className="text-gray-600 hover:text-gray-800"
                onClick={resetModalState}
              >
                <X size={20} />
              </button>
            </div>

            <h2 className="text-xl font-bold mb-4">Upload Files</h2>

            {/* File Upload Button */}
             <div className="flex justify-end mb-4 space-x-2">
               <button
                 className="px-4 py-2 border border-gray-300 rounded-md cursor-pointer bg-white text-gray-800 hover:bg-gray-50 font-semibold"
                 onClick={handleChooseFiles}
               >
                 + Choose Files
               </button>
               <input
                 type="file"
                 multiple
                 accept=".csv,.xlsm,.xlsx,.xls,.fdv,.r"
                 onChange={handleFileChange}
                 ref={fileInputRef}
                 className="hidden"
                 tabIndex={-1}
               />
               <button
                 className={`px-4 py-2 rounded-md font-semibold text-white ${isBatchUploading ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'} transition-colors`}
                 onClick={handleBatchUpload}
                 disabled={isBatchUploading || fileList.filter(f => f.status === 'pending').length === 0}
               >
                 {isBatchUploading ? 'Uploading...' : 'Upload All'}
               </button>
             </div>

             {/* Batch upload progress bar/prompt */}
             {isBatchUploading && (
               <div className="w-full flex items-center space-x-2 my-2">
                 <div className="flex-1 h-2 bg-gray-200 rounded">
                   <div
                     className="h-2 bg-blue-500 rounded"
                     style={{ width: `${(batchProgress.current / batchProgress.total) * 100}%` }}
                   ></div>
                 </div>
                <span className="text-sm text-gray-600 min-w-max">{` ${batchProgress.current} / ${batchProgress.total}`}</span>
               </div>
             )}

             {/* File table */}
             <div className="flex-grow overflow-y-auto">
                 <table className="w-full text-sm text-left border-separate border-spacing-0">
                 <thead className="bg-gray-100 text-gray-700 text-sm font-semibold sticky top-0">
                     <tr>
                     <th className="py-2 px-3 border-b border-gray-200 text-left w-1/3">File Name</th>
                     <th className="py-2 px-3 border-b border-gray-200 text-left">Type</th>
                     <th className="py-2 px-3 border-b border-gray-200 text-left">Area</th>
                     <th className="py-2 px-3 border-b border-gray-200 text-left">{fileList.some(f => f.type === 'Monitor Time Series Data') ? 'Interim' : ''}</th>
                     <th className="py-2 px-3 border-b border-gray-200 text-center">Status</th>
                     <th className="py-2 px-3 border-b border-gray-200 text-center">Edit</th>
                     </tr>
                 </thead>
                 <tbody>
                     {fileList.map((f) => (
                     <tr key={f.id} className="hover:bg-gray-50">
                         <td className="py-2 px-3 border-b border-gray-200 align-middle break-all">{f.name}</td>
                         <td className="py-2 px-3 border-b border-gray-200 align-middle">
                              <select
                                 value={f.type}
                                 onChange={(e) => handleMetadataChange(f.id, 'type', e.target.value)}
                                 className="block w-full text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                                 disabled={f.status === 'uploading' || f.status === 'success'}
                             >
                                 {FILE_TYPES.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                             </select>
                         </td>
                          <td className="py-2 px-3 border-b border-gray-200 align-middle">
                             <input
                                 type="text"
                                 value={f.area}
                                 onChange={(e) => handleMetadataChange(f.id, 'area', e.target.value)}
                                 placeholder="e.g., Central"
                                 className="block w-full shadow-sm sm:text-sm border-gray-300 rounded-md p-2"
                                 disabled={f.status === 'uploading' || f.status === 'success'}
                             />
                         </td>
                         {/* Interim input box, only shown for time series data type */}
                         <td className="py-2 px-3 border-b border-gray-200 align-middle">
                           {f.type === 'Monitor Time Series Data' ? (
                             <input
                               type="number"
                               min="1"
                               value={f.interim}
                               onChange={e => handleMetadataChange(f.id, 'interim', e.target.value.replace(/\D/g, ''))}
                               placeholder="Interim number (e.g., 1, 2, 3, 4)"
                               className="block w-full shadow-sm sm:text-sm border-gray-300 rounded-md p-2"
                               disabled={f.status === 'uploading' || f.status === 'success'}
                               required
                             />
                           ) : null}
                         </td>
                         <td className="py-2 px-3 border-b border-gray-200 text-center align-middle">
                         {f.status === 'pending' && <span className="text-gray-500">Pending</span>}
                         {f.status === 'uploading' && <span className="text-blue-600 font-semibold animate-pulse">Uploading...</span>}
                         {f.status === 'success' && <span className="text-green-600 font-semibold">Success</span>}
                         {f.status === 'error' && <span className="text-red-600 font-semibold" title={f.message}>Error</span>}
                         </td>
                         <td className="py-2 px-3 border-b border-gray-200 text-center align-middle">
                             <div className="flex justify-center items-center space-x-2">
                                 <button
                                     className="text-blue-600 hover:text-blue-800 disabled:text-gray-400 disabled:cursor-not-allowed"
                                     onClick={() => handleUploadFile(f.id)}
                                     title="Upload file"
                                     disabled={f.status === 'uploading' || f.status === 'success'}
                                 >
                                     <Upload size={18} />
                                 </button>
                                 <button
                                     className="text-red-500 hover:text-red-700 disabled:text-gray-400 disabled:cursor-not-allowed"
                                     onClick={() => handleRemoveFile(f.id)}
                                     title="Remove file"
                                     disabled={f.status === 'uploading'}
                                 >
                                     <X size={18} />
                                 </button>
                             </div>
                         </td>
                     </tr>
                     ))}
                 </tbody>
                 </table>
                 {fileList.length === 0 && (
                     <div className="text-center py-12 text-gray-500">
                         Please choose files to upload.
                     </div>
                 )}
             </div>
           </div>
         </div>
       )}

       {/* Batch type area fill popup */}
       {isConfirmModalOpen && (
         <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center" style={{zIndex: 9999}}>
           <div className="bg-white rounded-lg shadow-xl p-8 w-[400px] flex flex-col">
             <div className="flex justify-between items-center mb-4">
               <h3 className="text-lg font-bold">Batch Fill File Info</h3>
               <button className="text-gray-600 hover:text-gray-800" onClick={() => setIsConfirmModalOpen(false)}>
                 <X size={20} />
               </button>
             </div>
             <div className="mb-4 flex flex-col space-y-4">
               <div>
                 <label className="block text-sm font-medium mb-1">Type</label>
                 <select
                   value={batchType}
                   onChange={e => setBatchType(e.target.value)}
                   className="block w-full border-gray-300 rounded-md p-2"
                 >
                   {FILE_TYPES.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                 </select>
               </div>
               <div>
                 <label className="block text-sm font-medium mb-1">Area</label>
                 <input
                   type="text"
                   value={batchArea}
                   onChange={e => setBatchArea(e.target.value)}
                   placeholder="e.g., Central"
                   className="block w-full border-gray-300 rounded-md p-2"
                 />
               </div>
               {batchType === 'Monitor Time Series Data' && (
                 <div>
                   <label className="block text-sm font-medium mb-1">Interim</label>
                   <input
                     type="number"
                     min="1"
                     value={batchInterim}
                     onChange={e => setBatchInterim(e.target.value.replace(/\D/g, ''))}
                     placeholder="Interim number (e.g., 1, 2, 3, 4)"
                     className="block w-full border-gray-300 rounded-md p-2"
                   />
                 </div>
               )}
             </div>
             <div className="flex justify-end space-x-4 mt-4">
               <button
                 className="px-4 py-2 rounded-md font-semibold text-gray-700 bg-gray-200 hover:bg-gray-300"
                 onClick={() => setIsConfirmModalOpen(false)}
               >
                 Cancel
               </button>
               <button
                 className="px-4 py-2 rounded-md font-semibold text-white bg-blue-600 hover:bg-blue-700"
                 onClick={() => {
                   setIsConfirmModalOpen(false);
                   setTimeout(() => {
                     if (fileInputRef.current) fileInputRef.current.click();
                   }, 100);
                 }}
                 disabled={!batchArea || (batchType === 'Monitor Time Series Data' && !batchInterim)}
               >
                 Next
               </button>
             </div>
           </div>
         </div>
       )}
     </div>
   );
 }

 export default HomePage; 