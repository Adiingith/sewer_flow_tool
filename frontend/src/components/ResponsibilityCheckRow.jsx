import React from 'react';

const ResponsibilityCheckRow = React.memo(({ responsibility, onFieldChange }) => {
    const inputClass = "w-full border rounded p-1 text-sm";

    const handleInputChange = (field, value) => {
        onFieldChange(responsibility.monitor_id, field, value);
    };

    return (
        <tr key={responsibility.monitor_id}>
            <td className="px-4 py-2 font-medium text-sm">{responsibility.monitorName}</td>
            <td className="px-2 py-1">
                <input 
                    type="text" 
                    value={responsibility.action_type || ''} 
                    onChange={(e) => handleInputChange('action_type', e.target.value)} 
                    className={inputClass} 
                />
            </td>
            <td className="px-2 py-1">
                <input 
                    type="text" 
                    value={responsibility.requester || ''} 
                    onChange={(e) => handleInputChange('requester', e.target.value)} 
                    className={inputClass} 
                />
            </td>
            <td className="px-2 py-1">
                <input 
                    type="text" 
                    value={responsibility.removal_checker || ''} 
                    onChange={(e) => handleInputChange('removal_checker', e.target.value)} 
                    className={inputClass} 
                />
            </td>
            <td className="px-2 py-1">
                <input 
                    type="text" 
                    value={responsibility.removal_reviewer || ''} 
                    onChange={(e) => handleInputChange('removal_reviewer', e.target.value)} 
                    className={inputClass} 
                />
            </td>
            <td className="px-2 py-1">
                <input 
                    type="date" 
                    value={responsibility.removal_date ? new Date(responsibility.removal_date).toISOString().split('T')[0] : ''} 
                    onChange={(e) => handleInputChange('removal_date', e.target.value)} 
                    className={inputClass} 
                />
            </td>
        </tr>
    );
});

export default ResponsibilityCheckRow; 