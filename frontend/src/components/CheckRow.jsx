import React from 'react';

const CheckRow = React.memo(({ check, handleInputChange }) => {
  const inputClass = "mt-1 block w-full border border-gray-300 rounded-md shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm";
  const selectClass = "mt-1 block w-full pl-3 pr-10 py-2 text-base border border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md";

  return (
    <tr>
      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{check.monitor_name}</td>
      <td className="px-2 py-1">
        <input type="text" value={check.mh_reference || ''} onChange={(e) => handleInputChange(check.monitor_id, 'mh_reference', e.target.value)} className={inputClass} />
      </td>
      <td className="px-2 py-1">
        <input type="text" value={check.pipe || ''} onChange={(e) => handleInputChange(check.monitor_id, 'pipe', e.target.value)} className={inputClass} />
      </td>
      <td className="px-2 py-1">
        <input type="text" value={check.position || ''} onChange={(e) => handleInputChange(check.monitor_id, 'position', e.target.value)} className={inputClass} />
      </td>
      <td className="px-2 py-1">
        <select value={check.correct_location} onChange={(e) => handleInputChange(check.monitor_id, 'correct_location', e.target.value)} className={selectClass}>
          <option value="Y">Y</option>
          <option value="N">N</option>
        </select>
      </td>
      <td className="px-2 py-1">
        <select value={check.correct_install_pipe} onChange={(e) => handleInputChange(check.monitor_id, 'correct_install_pipe', e.target.value)} className={selectClass}>
          <option value="Y">Y</option>
          <option value="N">N</option>
        </select>
      </td>
      <td className="px-2 py-1">
        <select value={check.correct_pipe_size} onChange={(e) => handleInputChange(check.monitor_id, 'correct_pipe_size', e.target.value)} className={selectClass}>
          <option value="Y">Y</option>
          <option value="N">N</option>
        </select>
      </td>
      <td className="px-2 py-1">
        <select value={check.correct_pipe_shape} onChange={(e) => handleInputChange(check.monitor_id, 'correct_pipe_shape', e.target.value)} className={selectClass}>
          <option value="Y">Y</option>
          <option value="N">N</option>
        </select>
      </td>
      <td className="px-2 py-1">
        <textarea 
          value={check.comments} 
          onChange={(e) => handleInputChange(check.monitor_id, 'comments', e.target.value)} 
          rows="2" 
          className={inputClass}
          placeholder="Enter notes...">
        </textarea>
      </td>
    </tr>
  );
});

export default CheckRow; 