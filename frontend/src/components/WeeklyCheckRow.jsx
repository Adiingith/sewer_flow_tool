import React from 'react';
import { Trash2 } from 'lucide-react';

const WeeklyCheckRow = React.memo(({ check, handleInputChange, handleRemoveNewCheck, isNew }) => {
  const inputClass = "block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm";
  const readOnlyClass = "px-4 py-2 whitespace-nowrap bg-gray-50 text-gray-700";
  
  const commentsValue = (check.comments && typeof check.comments === 'object' && check.comments.notes !== undefined) ? check.comments.notes : (check.comments || '');
  const actionsValue = (check.actions && typeof check.actions === 'object' && check.actions.actions !== undefined) ? check.actions.actions : (check.actions || '');

  return (
    <tr className="bg-blue-50 hover:bg-blue-100">
      <td className={`${readOnlyClass} text-center`}>{check.interim}</td>
      <td className="px-4 py-1 align-top">
         <input 
          type="date" 
          value={check.check_date || ''} 
          onChange={(e) => handleInputChange(check.monitor_id, check.interim, 'check_date', e.target.value)} 
          className={inputClass}
        />
      </td>
      <td className="px-4 py-1 align-top">
        <input 
          type="number" 
          value={check.silt_mm || ''} 
          onChange={(e) => handleInputChange(check.monitor_id, check.interim, 'silt_mm', e.target.value)} 
          className={inputClass}
          placeholder="mm"
        />
      </td>
      <td className="px-4 py-1 align-top">
        <textarea 
          value={commentsValue || ''} 
          onChange={(e) => handleInputChange(check.monitor_id, check.interim, 'comments', e.target.value)} 
          rows="2" 
          className={`${inputClass} w-full`}
          placeholder="Enter notes...">
        </textarea>
      </td>
      <td className="px-4 py-1 align-top">
        <textarea 
          value={actionsValue || ''} 
          onChange={(e) => handleInputChange(check.monitor_id, check.interim, 'actions', e.target.value)} 
          rows="2" 
          className={`${inputClass} w-full`}
          placeholder="Enter actions...">
        </textarea>
      </td>
      <td className="px-4 py-1 align-top">
        <textarea 
          value={check.data_quality_check || ''} 
          onChange={(e) => handleInputChange(check.monitor_id, check.interim, 'data_quality_check', e.target.value)} 
          rows="2" 
          className={`${inputClass} w-full`}
          placeholder="Enter data quality check...">
        </textarea>
      </td>
      <td className="px-4 py-1 align-middle text-center">
        {isNew && (
          <button
            type="button"
            onClick={() => handleRemoveNewCheck(check.monitor_id, check.interim)}
            className="text-red-500 hover:text-red-700 p-1 rounded-full hover:bg-red-100"
            title="Cancel this new check"
          >
            <Trash2 size={18} />
          </button>
        )}
      </td>
    </tr>
  );
});

export default WeeklyCheckRow; 